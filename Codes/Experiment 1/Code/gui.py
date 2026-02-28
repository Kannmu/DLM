import customtkinter as ctk
import threading
import time
import tkinter as tk
from tkinter import messagebox
import serial.tools.list_ports
from umh_controller import UMHController
from experiment_logic import ExperimentLogic
import os
import sys

# Set appearance and theme
ctk.set_appearance_mode("Dark")  # Modern dark theme
ctk.set_default_color_theme("blue")


class ExperimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DLM Pilot Experiment Controller")
        self.geometry("1100x750")

        # Configure main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.controller = UMHController()
        
        # Ensure path is absolute or correct relative to execution
        if getattr(sys, 'frozen', False):
            # If frozen (PyInstaller), use the directory of the executable
            base_dir = os.path.dirname(sys.executable)
        else:
            # If running as script, go up two levels from Code/gui.py
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        data_dir = os.path.join(base_dir, "Data")
        
        # Ensure Data directory exists
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except Exception as e:
                print(f"Error creating data directory: {e}")
                
        self.logic = ExperimentLogic(data_dir)

        self.create_widgets()

        self.is_running_trial = False
        self.start_response_time = None

    def create_widgets(self):
        # Create Tabview with improved styling
        self.tabview = ctk.CTkTabview(self, width=1000, height=700)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.tab_connect = self.tabview.add("Connection & Setup")
        self.tab_experiment = self.tabview.add("Experiment")

        # Configure grid for Experiment tab (Left Sidebar, Right Content)
        self.tab_experiment.grid_columnconfigure(1, weight=1)
        self.tab_experiment.grid_rowconfigure(0, weight=1)

        # Configure grid for Connection tab (Center alignment)
        self.tab_connect.grid_columnconfigure(0, weight=1)

        self.setup_connect_tab()
        self.setup_experiment_tab()

    def setup_connect_tab(self):
        # Container Frame for centering
        container = ctk.CTkFrame(self.tab_connect, fg_color="transparent")
        container.grid(row=0, column=0, sticky="n", pady=20)

        # --- Device Connection Card ---
        card_conn = ctk.CTkFrame(
            container, corner_radius=15, border_width=1, border_color="gray30"
        )
        card_conn.pack(pady=10, padx=20, fill="x", ipadx=20, ipady=20)

        ctk.CTkLabel(
            card_conn,
            text="Device Connection",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(5, 15))

        # Connection Controls Row
        conn_row = ctk.CTkFrame(card_conn, fg_color="transparent")
        conn_row.pack(fill="x", pady=5)

        self.btn_scan = ctk.CTkButton(
            conn_row, text="↻ Scan Ports", command=self.scan_ports, width=120
        )
        self.btn_scan.pack(side="left", padx=5)

        self.combo_ports = ctk.CTkComboBox(
            conn_row, values=["No Ports Found"], width=250
        )
        self.combo_ports.pack(side="left", padx=5, fill="x", expand=True)

        self.btn_connect = ctk.CTkButton(
            card_conn,
            text="Connect Device",
            command=self.connect_device,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.btn_connect.pack(pady=20, fill="x", padx=5)

        self.lbl_status = ctk.CTkLabel(
            card_conn,
            text="Status: Disconnected",
            text_color="#FF5555",
            font=ctk.CTkFont(size=13),
        )
        self.lbl_status.pack(pady=(0, 5))

        # --- Demo Mapping Card ---
        card_map = ctk.CTkFrame(
            container, corner_radius=15, border_width=1, border_color="gray30"
        )
        card_map.pack(pady=20, padx=20, fill="both", expand=True, ipadx=20, ipady=20)

        ctk.CTkLabel(
            card_map,
            text="Stimulus Mapping (Demo Indices)",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(5, 10))

        self.btn_scan_demos = ctk.CTkButton(
            card_map,
            text="Scan Demos (0-10)",
            command=self.scan_demos,
            fg_color="gray40",
            hover_color="gray50",
        )
        self.btn_scan_demos.pack(pady=5)

        self.textbox_demos = ctk.CTkTextbox(
            card_map, height=200, font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.textbox_demos.pack(pady=10, fill="both", expand=True)
        self.textbox_demos.insert(
            "0.0", "Click 'Scan Demos' to find available stimuli on device.\n"
        )

    def setup_experiment_tab(self):
        # --- Left Panel: Controls Sidebar ---
        frame_left = ctk.CTkFrame(self.tab_experiment, width=300, corner_radius=10)
        frame_left.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=0)
        frame_left.grid_propagate(False)  # Maintain width

        ctk.CTkLabel(
            frame_left, text="Control Panel", font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(25, 30))

        # Participant ID Input
        p_frame = ctk.CTkFrame(frame_left, fg_color="transparent")
        p_frame.pack(fill="x", padx=20)
        ctk.CTkLabel(
            p_frame, text="Participant ID", font=ctk.CTkFont(size=14), anchor="w"
        ).pack(fill="x")
        self.entry_pid = ctk.CTkEntry(p_frame, placeholder_text="e.g. P01", height=35)
        self.entry_pid.pack(fill="x", pady=(5, 15))

        self.btn_start_exp = ctk.CTkButton(
            frame_left,
            text="Start New Experiment",
            command=self.start_experiment_flow,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            height=35,
        )
        self.btn_start_exp.pack(pady=5, padx=20, fill="x")

        self.btn_next_block = ctk.CTkButton(
            frame_left,
            text="Start Next Block",
            command=self.next_block_action,
            fg_color="#E59400",
            hover_color="#C27D00",
            text_color="white",
            height=35,
            state="disabled",
        )
        self.btn_next_block.pack(pady=5, padx=20, fill="x")

        # Separator
        ctk.CTkFrame(frame_left, height=2, fg_color=("gray70", "gray30")).pack(
            fill="x", padx=20, pady=25
        )

        # Block Info
        self.lbl_block_info = ctk.CTkLabel(
            frame_left,
            text="Block: - / -",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3B8ED0",
        )
        self.lbl_block_info.pack(pady=5)

        # Trial Status
        self.lbl_trial_info = ctk.CTkLabel(
            frame_left, text="Trial: - / -", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.lbl_trial_info.pack(pady=5)

        # Play Button Container
        self.frame_play_container = ctk.CTkFrame(frame_left, fg_color="transparent")
        self.frame_play_container.pack(pady=30, padx=20, fill="x")

        # Play Button (Prominent)
        self.btn_play_trial = ctk.CTkButton(
            self.frame_play_container,
            text="▶ PLAY SEQUENCE",
            command=self.play_trial_sequence,
            fg_color="#2CC985",
            hover_color="#229965",
            text_color="white",
            height=60,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.btn_play_trial.pack(fill="x")

        # Progress Bar
        ctk.CTkLabel(
            frame_left, text="Sequence Progress", font=ctk.CTkFont(size=12)
        ).pack(pady=(20, 5))
        self.progress_bar = ctk.CTkProgressBar(
            frame_left, height=12, progress_color="#2CC985"
        )
        self.progress_bar.pack(padx=20, fill="x")
        self.progress_bar.set(0)

        # --- Right Panel: Response Form ---
        frame_right = ctk.CTkFrame(self.tab_experiment, fg_color="transparent")
        frame_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            frame_right,
            text="Participant Response (2AFC)",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(15, 25))

        # Container for Questions (Keeps them above the submit button)
        self.frame_questions = ctk.CTkFrame(frame_right, fg_color="transparent")
        self.frame_questions.pack(fill="x", pady=10)

        # Question 1 Card (Intensity)
        self.q1_card = ctk.CTkFrame(self.frame_questions, corner_radius=10)
        self.q1_card.pack(fill="x", pady=10, ipadx=15, ipady=15)

        ctk.CTkLabel(
            self.q1_card,
            text="1. Intensity",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3B8ED0",
        ).pack(anchor="w", padx=10)
        ctk.CTkLabel(
            self.q1_card,
            text="Which stimulus felt STRONGER?",
            font=ctk.CTkFont(size=15),
        ).pack(anchor="w", padx=10, pady=(2, 10))

        self.var_intensity = tk.StringVar(value="")
        q1_opts = ctk.CTkFrame(self.q1_card, fg_color="transparent")
        q1_opts.pack(fill="x", padx=20)

        self.rb_int_a = ctk.CTkRadioButton(
            q1_opts,
            text="Stimulus A (First)",
            variable=self.var_intensity,
            value="A",
            font=ctk.CTkFont(size=14),
        )
        self.rb_int_a.pack(side="left", padx=30, pady=10)
        self.rb_int_b = ctk.CTkRadioButton(
            q1_opts,
            text="Stimulus B (Second)",
            variable=self.var_intensity,
            value="B",
            font=ctk.CTkFont(size=14),
        )
        self.rb_int_b.pack(side="left", padx=30, pady=10)

        # Question 2 Card (Clarity)
        self.q2_card = ctk.CTkFrame(self.frame_questions, corner_radius=10)
        self.q2_card.pack(fill="x", pady=10, ipadx=15, ipady=15)

        ctk.CTkLabel(
            self.q2_card,
            text="2. Clarity / Size",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3B8ED0",
        ).pack(anchor="w", padx=10)
        ctk.CTkLabel(
            self.q2_card,
            text="Which stimulus felt SMALLER / CLEARER?",
            font=ctk.CTkFont(size=15),
        ).pack(anchor="w", padx=10, pady=(2, 10))

        self.var_clarity = tk.StringVar(value="")
        q2_opts = ctk.CTkFrame(self.q2_card, fg_color="transparent")
        q2_opts.pack(fill="x", padx=20)

        self.rb_clr_a = ctk.CTkRadioButton(
            q2_opts,
            text="Stimulus A (First)",
            variable=self.var_clarity,
            value="A",
            font=ctk.CTkFont(size=14),
        )
        self.rb_clr_a.pack(side="left", padx=30, pady=10)
        self.rb_clr_b = ctk.CTkRadioButton(
            q2_opts,
            text="Stimulus B (Second)",
            variable=self.var_clarity,
            value="B",
            font=ctk.CTkFont(size=14),
        )
        self.rb_clr_b.pack(side="left", padx=30, pady=10)

        # Submit Button Container
        self.frame_record_container = ctk.CTkFrame(frame_right, fg_color="transparent")
        self.frame_record_container.pack(pady=30, fill="x", padx=50)

        # Submit Button
        self.btn_record = ctk.CTkButton(
            self.frame_record_container,
            text="Submit Response & Next Trial",
            command=self.record_response,
            height=55,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        # Initially hidden until sequence played? Or visible?
        # User wants mutual exclusion. So usually hidden at start of trial.
        # But for safety, let's leave it packed here, and update_trial_display will handle visibility.
        self.btn_record.pack(fill="x")

        # Status Footer
        self.lbl_status_exp = ctk.CTkLabel(
            frame_right,
            text="Ready to start experiment",
            text_color="gray",
            font=ctk.CTkFont(size=13, slant="italic"),
        )
        self.lbl_status_exp.pack(side="bottom", pady=10)

    def scan_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.combo_ports.configure(values=ports)
            self.combo_ports.set(ports[0])
        else:
            self.combo_ports.configure(values=["No Ports"])
            self.combo_ports.set("No Ports")

    def connect_device(self):
        port = self.combo_ports.get()
        if port == "No Ports" or not port:
            return

        try:
            self.controller.connect(port)
            if self.controller.ping():
                self.lbl_status.configure(
                    text=f"Status: Connected to {port}", text_color="#2CC985"
                )  # Green
                messagebox.showinfo(
                    "Success", "Device connected and pinged successfully."
                )
            else:
                self.lbl_status.configure(
                    text="Status: Connected but Ping failed", text_color="orange"
                )
        except Exception as e:
            self.lbl_status.configure(text=f"Error: {e}", text_color="#FF5555")  # Red

    def scan_demos(self):
        if not self.controller.is_connected:
            messagebox.showerror("Error", "Connect device first.")
            return

        self.textbox_demos.delete("0.0", "end")
        self.textbox_demos.insert("end", "Scanning demos 0-9...\n")

        def run_scan():
            found_map = {}
            for i in range(10):
                try:
                    name = self.controller.set_demo(i)
                    if name:
                        self.after(
                            0,
                            lambda t=f"Index {i}: {name}\n": self.textbox_demos.insert(
                                "end", t
                            ),
                        )
                        # Try to map to known stimuli
                        for stim in self.logic.STIMULI:
                            if stim.lower() in name.lower():  # Simple matching
                                found_map[stim] = i
                    else:
                        self.after(
                            0,
                            lambda t=f"Index {i}: No Response/Error\n": self.textbox_demos.insert(
                                "end", t
                            ),
                        )
                except Exception as e:
                    self.after(
                        0,
                        lambda t=f"Index {i}: Exception {e}\n": self.textbox_demos.insert(
                            "end", t
                        ),
                    )

            # Disable output after scanning to ensure silence
            try:
                self.controller.enable_output(False)
            except Exception as e:
                print(f"Error disabling output after scan: {e}")

            self.logic.stimulus_indices = found_map
            self.after(
                0,
                lambda t=f"\nAuto-mapped: {found_map}\n": self.textbox_demos.insert(
                    "end", t
                ),
            )

        threading.Thread(target=run_scan, daemon=True).start()

    def start_experiment_flow(self):
        pid = self.entry_pid.get()
        if not pid:
            messagebox.showwarning("Warning", "Enter Participant ID")
            return

        self.logic.start_experiment(pid)
        self.btn_start_exp.configure(state="disabled")
        self.update_ui_for_block()
        messagebox.showinfo(
            "Info",
            f"Experiment Started.\nBlock 1: {self.logic.get_current_block_type()}",
        )

    def next_block_action(self):
        if self.logic.next_block():
            self.btn_next_block.configure(state="disabled")
            self.update_ui_for_block()
            messagebox.showinfo(
                "Info",
                f"Starting Block {self.logic.current_block_index + 1}: {self.logic.get_current_block_type()}",
            )
        else:
            messagebox.showinfo("Info", "Experiment Complete!")

    def update_ui_for_block(self):
        block_type = self.logic.get_current_block_type()
        idx = self.logic.current_block_index + 1
        total_blocks = len(self.logic.block_order)

        self.lbl_block_info.configure(
            text=f"Block: {idx} / {total_blocks} ({block_type})"
        )

        # Show/Hide cards
        if block_type == "Intensity":
            self.q1_card.pack(fill="x", pady=10, ipadx=15, ipady=15)
            self.q2_card.pack_forget()
        elif block_type == "Spatial":
            self.q1_card.pack_forget()
            self.q2_card.pack(fill="x", pady=10, ipadx=15, ipady=15)

        self.update_trial_display()

    def update_trial_display(self):
        current = self.logic.get_current_trial()
        idx = self.logic.current_trial_index + 1
        total = len(self.logic.trials)

        if current:
            self.lbl_trial_info.configure(text=f"Trial: {idx} / {total}")
            self.lbl_status_exp.configure(text=f"Current Pair: Hidden (Blind)")
            print(f"Trial {idx}: A={current[0]}, B={current[1]}")

            # Reset buttons state: Show Play, Hide Submit
            self.btn_play_trial.pack(fill="x")
            self.btn_play_trial.configure(state="normal", text="▶ PLAY SEQUENCE")

            self.btn_record.pack_forget()
            self.btn_record.configure(state="normal")
        else:
            self.lbl_trial_info.configure(text="Block Complete")
            self.lbl_status_exp.configure(text="All trials finished.")
            self.btn_play_trial.configure(state="disabled")
            self.btn_record.configure(state="disabled")
            # Ensure both visible but disabled? Or hide?
            # Let's keep them disabled so user sees "Game Over" state
            self.btn_play_trial.pack(fill="x")
            self.btn_record.pack(fill="x")

            if not self.logic.is_experiment_complete():
                self.btn_next_block.configure(state="normal")
            else:
                self.lbl_status_exp.configure(text="Experiment Finished.")
                self.btn_start_exp.configure(state="normal")

    def play_trial_sequence(self):
        if self.is_running_trial:
            return

        # Reset RT timer at start of playback
        self.start_response_time = None

        current = self.logic.get_current_trial()
        if not current:
            messagebox.showinfo("Info", "Block complete or not started.")
            return

        stim_a_name, stim_b_name = current

        # Get indices
        idx_a = self.logic.stimulus_indices.get(stim_a_name)
        idx_b = self.logic.stimulus_indices.get(stim_b_name)

        if idx_a is None or idx_b is None:
            messagebox.showerror(
                "Error",
                f"Stimulus indices not mapped. Please Scan Demos first.\nMissing: {stim_a_name if idx_a is None else ''} {stim_b_name if idx_b is None else ''}",
            )
            return

        self.is_running_trial = True
        self.btn_play_trial.configure(state="disabled", text="Running Sequence...")
        self.progress_bar.set(0)

        def run_sequence():
            try:
                # 1. Play A
                self.after(
                    0,
                    lambda: self.lbl_status_exp.configure(text="Playing Stimulus A..."),
                )
                self.controller.set_demo(idx_a)
                self.controller.enable_output(True)
                time.sleep(1.5)
                self.controller.enable_output(False)

                # 2. Pause
                self.after(
                    0, lambda: self.lbl_status_exp.configure(text="Pause (1.0s)...")
                )
                self.after(0, lambda: self.progress_bar.set(0.5))
                time.sleep(1.0)

                # 3. Play B
                self.after(
                    0,
                    lambda: self.lbl_status_exp.configure(text="Playing Stimulus B..."),
                )
                self.controller.set_demo(idx_b)
                self.controller.enable_output(True)
                time.sleep(1.5)
                self.controller.enable_output(False)

                # Start timing reaction time immediately after stimulus B ends
                self.start_response_time = time.time()

                self.after(
                    0,
                    lambda: self.lbl_status_exp.configure(
                        text="Waiting for response..."
                    ),
                )
                self.after(0, lambda: self.progress_bar.set(1.0))

            except Exception as e:
                self.after(
                    0,
                    lambda m=str(e): messagebox.showerror(
                        "Error", f"Sequence error: {m}"
                    ),
                )
            finally:
                self.is_running_trial = False

                # Sequence done: Hide Play, Show Submit
                def on_sequence_done():
                    self.btn_play_trial.pack_forget()
                    self.btn_record.pack(fill="x")
                    self.lbl_status_exp.configure(
                        text="Sequence finished. Please submit response."
                    )

                self.after(0, on_sequence_done)

        threading.Thread(target=run_sequence, daemon=True).start()

    def record_response(self):
        current = self.logic.get_current_trial()
        if not current:
            return

        block_type = self.logic.get_current_block_type()

        chosen_int = None
        chosen_clr = None

        if block_type == "Intensity":
            choice = self.var_intensity.get()
            if not choice:
                messagebox.showwarning(
                    "Warning", "Please select an option for Intensity."
                )
                return
            stim_a, stim_b = current
            chosen_int = stim_a if choice == "A" else stim_b

        elif block_type == "Spatial":
            choice = self.var_clarity.get()
            if not choice:
                messagebox.showwarning(
                    "Warning", "Please select an option for Clarity."
                )
                return
            stim_a, stim_b = current
            chosen_clr = stim_a if choice == "A" else stim_b

        # Calculate RT
        rt = 0
        if self.start_response_time is not None:
            rt = time.time() - self.start_response_time
            # Reset after recording so subsequent clicks (if any) don't reuse old time
            self.start_response_time = None

        try:
            self.logic.save_trial_data(
                current[0], current[1], chosen_int, chosen_clr, rt
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")
            return

        # Reset selection
        self.var_intensity.set("")
        self.var_clarity.set("")

        # Next trial
        if self.logic.next_trial():
            self.update_trial_display()
        else:
            self.update_trial_display()
            messagebox.showinfo("Done", "Block finished!")


if __name__ == "__main__":
    app = ExperimentApp()
    app.mainloop()
