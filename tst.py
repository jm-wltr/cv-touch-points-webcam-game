import cv2

print("Testing camera access on Windows...")
print("-" * 50)

# Test multiple camera indices
for i in range(5):
    print(f"\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i} WORKS!")
            print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            # Show a test frame
            cv2.imshow(f'Camera {i} Test', frame)
            print(f"  Showing test window. Press any key to continue...")
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        else:
            print(f"✗ Camera {i} opened but couldn't read frame")
        cap.release()
    else:
        print(f"✗ Camera {i} failed to open")

print("\n" + "-" * 50)
print("Camera test complete!")

# Try DirectShow backend (Windows specific)
print("\nTrying with DirectShow backend...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✓ DirectShow backend WORKS!")
        cv2.imshow('DirectShow Test', frame)
        print("Press any key to close...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    cap.release()
else:
    print("✗ DirectShow backend failed")

print("\nIf a camera worked, note its index number!")