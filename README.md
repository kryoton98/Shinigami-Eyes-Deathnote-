# 👁️ Shinigami Surveillance Terminal

A high-performance, interactive computer vision photo booth inspired by the *Death Note* anime. 

This terminal uses spatial tracking and facial recognition to scan a room, lock onto faces, and overlay random "True Names" and "Lifespans" in real-time. It features a dual-screen web dashboard, automated ambient soundboards, and a custom "Night Vision" engine to cut through heavy party lighting.

## ✨ Features
* **Live Soul Tracking:** Automatically detects faces and assigns permanent Death Note characters (and random lifespans) to guests for the duration of the session.
* **Max-FPS Spatial Engine:** Uses a lightweight spatial tracking algorithm to keep text glued to moving faces at 30+ FPS, while farming out heavy AI recognition to background threads.
* **Night Vision (CLAHE):** Built-in contrast equalization strips away heavy color casts (like blue party LEDs) to pull facial features out of the shadows.
* **The Hacker Dashboard:** A sleek, dark-mode web UI featuring a live camera feed and an interactive "Signal Intercept" sidebar.
* **Dual-Capture System:** Take standard, clean photos or "Shinigami Vision" photos with the UI overlay applied directly to the image.
* **Ambient Soundboard:** Automatically plays eerie background audio (Ryuk's laugh, L's notification chime) at random intervals to set the mood in the room.

## 📁 Directory Structure
To run this project, your folder must look exactly like this:

```text
shinigami-eyes/
 ├── app.py                 # The main application script
 ├── .gitignore             # Keeps your captured photos off GitHub
 ├── custom_font.ttf        # Auto-downloads on first run
 ├── captures/              # Auto-generated folder for saved photos
 └── static/                # YOU MUST CREATE THIS FOLDER AND ADD THESE FILES
      ├── L_logo.png
      ├── Kira_logo.png
      ├── shutter.mp3
      ├── laugh.mp3
      └── chime.mp3

# This was orinally made for a party for my RP hall day! 