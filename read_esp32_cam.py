import serial
import pygame
import io
import sys

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/cu.usbmodem5B5F0214411' # Update this!
BAUD_RATE = 2000000

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.display.set_caption("ESP32-S3 Camera Stream")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit()

def read_frame():
    # 1. Hunt for the Sync Marker (0xAA 0xBB 0xCC 0xDD)
    sync_buffer = bytearray()
    while True:
        byte = ser.read(1)
        if not byte:
            return None # Timeout
        
        sync_buffer.append(byte[0])
        if len(sync_buffer) > 4:
            sync_buffer.pop(0)
            
        if sync_buffer == bytearray([0xAA, 0xBB, 0xCC, 0xDD]):
            break

    # 2. Read the 4-byte length header
    length_bytes = ser.read(4)
    if len(length_bytes) != 4:
        return None
    
    image_length = int.from_bytes(length_bytes, byteorder='little')

    # 3. Read the exact number of bytes for the image
    image_data = ser.read(image_length)
    if len(image_data) != image_length:
        return None

    return image_data

# Main Loop
running = True
print("Waiting for video feed...")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Grab the raw bytes from the serial port
    raw_jpeg = read_frame()
    if raw_jpeg:
        try:
            # Convert raw bytes into a Pygame surface
            image_stream = io.BytesIO(raw_jpeg)
            surface = pygame.image.load(image_stream)

            # Draw it to the window
            screen.blit(surface, (0, 0))
            pygame.display.flip()

        except pygame.error:
            # Occasionally a frame might drop or corrupt, just skip it
            pass

ser.close()
pygame.quit()