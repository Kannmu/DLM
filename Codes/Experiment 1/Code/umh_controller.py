import serial
import serial.tools.list_ports
import time
import struct
import threading

class UMHController:
    FRAME_HEADER = b'\xAA\x55'
    FRAME_TAIL = b'\x0D\x0A'
    
    CMD_ENABLE_DISABLE = 0x01
    CMD_PING = 0x02
    CMD_SET_DEMO = 0x07
    
    RSP_ACK = 0x80
    RSP_PING_ACK = 0x82
    RSP_DEMO_ACK = 0x86
    
    def __init__(self):
        self.ser = None
        self.is_connected = False
        self.lock = threading.Lock()

    def find_device(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            try:
                # Try to connect and ping
                print(f"Trying {port.device}...")
                self.connect(port.device)
                if self.ping():
                    print(f"Found UMH Device on {port.device}")
                    return port.device
                self.disconnect()
            except Exception as e:
                print(f"Error checking {port.device}: {e}")
                self.disconnect()
        return None

    def connect(self, port):
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        self.ser = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )
        self.is_connected = True

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False

    def _calculate_checksum(self, cmd_type, data_length, data):
        checksum = cmd_type + data_length
        for byte in data:
            checksum += byte
        return checksum & 0xFF

    def _send_frame(self, cmd_type, data=None):
        if not self.is_connected:
            raise Exception("Device not connected")
            
        if data is None:
            data = []
            
        data_length = len(data)
        checksum = self._calculate_checksum(cmd_type, data_length, data)
        
        frame = bytearray()
        frame.extend(self.FRAME_HEADER)
        frame.append(cmd_type)
        frame.append(data_length)
        frame.extend(data)
        frame.append(checksum)
        frame.extend(self.FRAME_TAIL)
        
        with self.lock:
            self.ser.write(frame)
            # time.sleep(0.01) # Small delay to ensure transmission

    def _read_response(self, expected_cmd_type=None, timeout=1.0):
        if not self.is_connected:
            return None
            
        start_time = time.time()
        buffer = bytearray()
        
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                buffer.extend(self.ser.read(self.ser.in_waiting))
                
                # Search for header
                while len(buffer) >= 6: # Min frame size: Header(2) + Cmd(1) + Len(1) + Checksum(1) + Tail(2) = 7. Wait, Len=0 -> Data=0. So 2+1+1+0+1+2 = 7 bytes.
                    if buffer[0:2] != self.FRAME_HEADER:
                        buffer.pop(0)
                        continue
                    
                    if len(buffer) < 4:
                        break
                        
                    data_len = buffer[3]
                    total_len = 2 + 1 + 1 + data_len + 1 + 2
                    
                    if len(buffer) < total_len:
                        break # Wait for more data
                        
                    # Extract frame
                    frame = buffer[:total_len]
                    cmd_type = frame[2]
                    data = frame[4:4+data_len]
                    received_checksum = frame[4+data_len]
                    tail = frame[4+data_len+1:]
                    
                    # Verify checksum
                    calc_checksum = self._calculate_checksum(cmd_type, data_len, data)
                    
                    if calc_checksum == received_checksum and tail == self.FRAME_TAIL:
                        # Valid frame
                        buffer = buffer[total_len:] # Remove processed frame
                        if expected_cmd_type is None or cmd_type == expected_cmd_type:
                            return cmd_type, data
                    else:
                        # Invalid checksum or tail, remove header and continue search
                        buffer.pop(0)
                        
            time.sleep(0.005)
        return None

    def ping(self):
        try:
            random_data = [0x12, 0x34, 0x56]
            self._send_frame(self.CMD_PING, random_data)
            response = self._read_response(self.RSP_PING_ACK)
            if response:
                cmd, data = response
                return list(data) == random_data
            return False
        except Exception:
            return False

    def set_demo(self, demo_index):
        """
        Set demo mode.
        Returns the name of the demo if successful, None otherwise.
        """
        self._send_frame(self.CMD_SET_DEMO, [demo_index])
        response = self._read_response() # Can be RSP_DEMO_ACK or RSP_ERROR_CODE
        
        if response:
            cmd, data = response
            if cmd == self.RSP_DEMO_ACK:
                try:
                    return data.decode('utf-8').strip('\x00')
                except:
                    return "Unknown (Decode Error)"
        return None

    def enable_output(self, enable=True):
        val = 1 if enable else 0
        self._send_frame(self.CMD_ENABLE_DISABLE, [val])
        response = self._read_response()
        if response:
            cmd, _ = response
            return cmd == self.RSP_ACK
        return False

if __name__ == "__main__":
    # Test stub
    ctl = UMHController()
    print("Searching for device...")
    port = ctl.find_device()
    if port:
        print(f"Connected to {port}")
        print(f"Ping: {ctl.ping()}")
        print(f"Enable: {ctl.enable_output(True)}")
        time.sleep(1)
        print(f"Disable: {ctl.enable_output(False)}")
        print(f"Set Demo 0: {ctl.set_demo(0)}")
        ctl.disconnect()
    else:
        print("No device found.")
