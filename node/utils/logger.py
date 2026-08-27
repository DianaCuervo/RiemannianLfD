import sys
import os
import datetime


class ConsoleLogger:
    """
    Buffers early console output in memory, then writes it to a file
    once the file path is dynamically generated.
    """

    def __init__(self):
        self.terminal = sys.stdout
        self.log_file = None
        self.buffer = []  # Stores early prints in RAM

    def set_log_file(self, save_dir, model_name):
        """Creates the file and dumps everything printed so far."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as NODE_lasa_NShape_RiemannianMSE_EGI_20260824_153022.txt
        log_path = os.path.join(save_dir, f"{model_name}_Log_{timestamp}.txt")

        self.log_file = open(log_path, "a", encoding="utf-8")
        self.terminal.write(f"\n📝 Log file generated! Saving all output to: {log_path}\n\n")

        # Dump the memory buffer into the text file!
        if self.buffer:
            self.log_file.write("".join(self.buffer))
            self.buffer = []  # Clear memory
            self.log_file.flush()

    def write(self, message):
        # 1. Always print to the real terminal
        self.terminal.write(message)

        # 2. If the file is ready, write to it. If not, save it in the buffer!
        if self.log_file is not None:
            self.log_file.write(message)
            self.log_file.flush()
        else:
            self.buffer.append(message)

    def flush(self):
        self.terminal.flush()
        if self.log_file is not None:
            self.log_file.flush()