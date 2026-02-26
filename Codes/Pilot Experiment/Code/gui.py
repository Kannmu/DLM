import customtkinter as ctk
import threading
import time
import tkinter as tk
from tkinter import messagebox
import serial.tools.list_ports
from umh_controller import UMHController
from experiment_logic import ExperimentLogic

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ExperimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("DLM Pilot Experiment Controller")
        self.geometry("900x650")
        
        self.controller = UMHController()
        # Ensure path is absolute or correct relative to execution
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "Data")
        self.logic = ExperimentLogic(data_dir)
        
        self.create_widgets()
        
        self.is_running_trial = False
        self.start_response_time = None
        
        # Periodic check for connection status if needed
        # self.after(1000, self.check_connection)

    def create_widgets(self):
        # Create Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.tab_connect = self.tabview.add("Connection & Setup")
        self.tab_experiment = self.tabview.add("Experiment")
        
        self.setup_connect_tab()
        self.setup_experiment_tab()

    def setup_connect_tab(self):
        # Connection Frame
        frame_conn = ctk.CTkFrame(self.tab_connect)
        frame_conn.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(frame_conn, text="Device Connection", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.btn_scan = ctk.CTkButton(frame_conn, text="Scan Ports", command=self.scan_ports)
        self.btn_scan.pack(pady=5)
        
        self.combo_ports = ctk.CTkComboBox(frame_conn, values=["No Ports Found"])
        self.combo_ports.pack(pady=5)
        
        self.btn_connect = ctk.CTkButton(frame_conn, text="Connect", command=self.connect_device)
        self.btn_connect.pack(pady=5)
        
        self.lbl_status = ctk.CTkLabel(frame_conn, text="Status: Disconnected", text_color="red")
        self.lbl_status.pack(pady=5)
        
        # Demo Mapping Frame
        frame_map = ctk.CTkFrame(self.tab_connect)
        frame_map.pack(pady=20, padx=20, fill="both", expand=True)
        
        ctk.CTkLabel(frame_map, text="Stimulus Mapping (Demo Indices)", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.btn_scan_demos = ctk.CTkButton(frame_map, text="Scan Demos (0-10)", command=self.scan_demos)
        self.btn_scan_demos.pack(pady=5)
        
        self.textbox_demos = ctk.CTkTextbox(frame_map, height=200)
        self.textbox_demos.pack(pady=5, fill="both", expand=True)
        self.textbox_demos.insert("0.0", "Click 'Scan Demos' to find available stimuli on device.\n")

    def setup_experiment_tab(self):
        # Left Panel: Controls
        frame_left = ctk.CTkFrame(self.tab_experiment, width=300)
        frame_left.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(frame_left, text="Experiment Control", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.entry_pid = ctk.CTkEntry(frame_left, placeholder_text="Participant ID")
        self.entry_pid.pack(pady=5)
        
        self.btn_start_block = ctk.CTkButton(frame_left, text="Generate New Block", command=self.start_block)
        self.btn_start_block.pack(pady=5)
        
        self.lbl_trial_info = ctk.CTkLabel(frame_left, text="Trial: 0 / 20", font=("Arial", 14))
        self.lbl_trial_info.pack(pady=20)
        
        self.btn_play_trial = ctk.CTkButton(frame_left, text="▶ Play Trial Sequence", command=self.play_trial_sequence, fg_color="green", height=40)
        self.btn_play_trial.pack(pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(frame_left)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Right Panel: Response
        frame_right = ctk.CTkFrame(self.tab_experiment)
        frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(frame_right, text="Participant Response (2AFC)", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Question 1
        ctk.CTkLabel(frame_right, text="1. Intensity: Which was Stronger?", font=("Arial", 14)).pack(pady=5)
        self.var_intensity = tk.StringVar(value="")
        self.rb_int_a = ctk.CTkRadioButton(frame_right, text="Stimulus A (First)", variable=self.var_intensity, value="A")
        self.rb_int_a.pack(pady=2)
        self.rb_int_b = ctk.CTkRadioButton(frame_right, text="Stimulus B (Second)", variable=self.var_intensity, value="B")
        self.rb_int_b.pack(pady=2)
        
        # Question 2
        ctk.CTkLabel(frame_right, text="2. Clarity: Which was Smaller/Clearer?", font=("Arial", 14)).pack(pady=15)
        self.var_clarity = tk.StringVar(value="")
        self.rb_clr_a = ctk.CTkRadioButton(frame_right, text="Stimulus A (First)", variable=self.var_clarity, value="A")
        self.rb_clr_a.pack(pady=2)
        self.rb_clr_b = ctk.CTkRadioButton(frame_right, text="Stimulus B (Second)", variable=self.var_clarity, value="B")
        self.rb_clr_b.pack(pady=2)
        
        self.btn_record = ctk.CTkButton(frame_right, text="Record Response & Next", command=self.record_response, height=40)
        self.btn_record.pack(pady=20)
        
        self.lbl_status_exp = ctk.CTkLabel(frame_right, text="Ready", text_color="gray", font=("Arial", 12))
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
                self.lbl_status.configure(text=f"Connected to {port}", text_color="green")
                messagebox.showinfo("Success", "Device connected and pinged successfully.")
            else:
                self.lbl_status.configure(text="Connected but Ping failed", text_color="orange")
        except Exception as e:
            self.lbl_status.configure(text=f"Error: {e}", text_color="red")

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
                        self.after(0, lambda t=f"Index {i}: {name}\n": self.textbox_demos.insert("end", t))
                        # Try to map to known stimuli
                        for stim in self.logic.STIMULI:
                            if stim.lower() in name.lower(): # Simple matching
                                found_map[stim] = i
                    else:
                        self.after(0, lambda t=f"Index {i}: No Response/Error\n": self.textbox_demos.insert("end", t))
                except Exception as e:
                    self.after(0, lambda t=f"Index {i}: Exception {e}\n": self.textbox_demos.insert("end", t))
            
            # Disable output after scanning to ensure silence
            try:
                self.controller.enable_output(False)
            except Exception as e:
                print(f"Error disabling output after scan: {e}")

            self.logic.stimulus_indices = found_map
            self.after(0, lambda t=f"\nAuto-mapped: {found_map}\n": self.textbox_demos.insert("end", t))
            
        threading.Thread(target=run_scan, daemon=True).start()

    def start_block(self):
        pid = self.entry_pid.get()
        if not pid:
            messagebox.showwarning("Warning", "Enter Participant ID")
            return
            
        self.logic.set_participant(pid)
        self.logic.generate_sequence()
        self.update_trial_display()
        messagebox.showinfo("Info", "New block generated. Ready to start.")

    def update_trial_display(self):
        current = self.logic.get_current_trial()
        idx = self.logic.current_trial_index + 1
        total = len(self.logic.trials)
        
        if current:
            self.lbl_trial_info.configure(text=f"Trial: {idx} / {total}")
            self.lbl_status_exp.configure(text=f"Current Pair: Hidden (Blind)")
            print(f"Trial {idx}: A={current[0]}, B={current[1]}")
        else:
            self.lbl_trial_info.configure(text="Block Complete")
            self.lbl_status_exp.configure(text="All trials finished.")

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
            messagebox.showerror("Error", f"Stimulus indices not mapped. Please Scan Demos first.\nMissing: {stim_a_name if idx_a is None else ''} {stim_b_name if idx_b is None else ''}")
            return
            
        self.is_running_trial = True
        self.btn_play_trial.configure(state="disabled")
        self.progress_bar.set(0)
        
        def run_sequence():
            try:
                # 1. Play A
                self.after(0, lambda: self.lbl_status_exp.configure(text="Playing Stimulus A..."))
                self.controller.set_demo(idx_a)
                self.controller.enable_output(True)
                time.sleep(1.5)
                self.controller.enable_output(False)
                
                # 2. Pause
                self.after(0, lambda: self.lbl_status_exp.configure(text="Pause (1.0s)..."))
                self.after(0, lambda: self.progress_bar.set(0.5))
                time.sleep(1.0)
                
                # 3. Play B
                self.after(0, lambda: self.lbl_status_exp.configure(text="Playing Stimulus B..."))
                self.controller.set_demo(idx_b)
                self.controller.enable_output(True)
                time.sleep(1.5)
                self.controller.enable_output(False)
                
                # Start timing reaction time immediately after stimulus B ends
                self.start_response_time = time.time()
                
                self.after(0, lambda: self.lbl_status_exp.configure(text="Waiting for response..."))
                self.after(0, lambda: self.progress_bar.set(1.0))
                
            except Exception as e:
                self.after(0, lambda m=str(e): messagebox.showerror("Error", f"Sequence error: {m}"))
            finally:
                self.is_running_trial = False
                self.after(0, lambda: self.btn_play_trial.configure(state="normal"))
        
        threading.Thread(target=run_sequence, daemon=True).start()

    def record_response(self):
        current = self.logic.get_current_trial()
        if not current:
            return
            
        int_choice = self.var_intensity.get()
        clr_choice = self.var_clarity.get()
        
        if not int_choice or not clr_choice:
            messagebox.showwarning("Warning", "Please select both options.")
            return
            
        stim_a, stim_b = current
        chosen_int_stim = stim_a if int_choice == "A" else stim_b
        chosen_clr_stim = stim_a if clr_choice == "A" else stim_b
        
        # Calculate RT
        rt = 0
        if self.start_response_time is not None:
            rt = time.time() - self.start_response_time
            # Reset after recording so subsequent clicks (if any) don't reuse old time
            self.start_response_time = None
        
        try:
            self.logic.save_trial_data(stim_a, stim_b, chosen_int_stim, chosen_clr_stim, rt)
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
