import cv2

ascii_chars = "@%#*+=-:. "

def pixel_to_ascii(pixel):
    return ascii_chars[pixel * len(ascii_chars) // 256]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize for terminal
    height, width = gray.shape
    new_width = 120
    new_height = int(height * new_width / width / 2)

    resized = cv2.resize(gray, (new_width, new_height))

    ascii_image = ""
    for row in resized:
        for pixel in row:
            ascii_image += pixel_to_ascii(pixel)
        ascii_image += "\n"

    print("\033c", end="")  # clear terminal
    print(ascii_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()