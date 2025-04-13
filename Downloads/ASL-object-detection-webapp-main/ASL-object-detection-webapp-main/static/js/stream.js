const img = document.getElementById("streamedImage");
const video = document.createElement("video");
const startButton = document.getElementById("startButton");
const stopWebcamButton = document.getElementById("stopWebcamButton");
let stream;
let sendFrames = true; // Flag to control frame sending

startButton.addEventListener("click", () => {
  if (!sendFrames) {
    sendFrames = true; // Set sendFrames flag to true when starting the webcam again
  }
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((str) => {
      stream = str; // Store the stream reference
      video.srcObject = stream;
      video.play();
      startSendingFrames();
    })
    .catch((error) => {
      console.error("Error accessing webcam:", error);
    });
});

stopWebcamButton.addEventListener("click", () => {
  stopWebcam();
});

function startSendingFrames() {
  const sendFrame = () => {
    if (!sendFrames) return; // Stop sending frames if sendFrames flag is false
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameData = canvas.toDataURL("image/jpeg");

    fetch("/video_feed", {
      method: "POST",
      body: new URLSearchParams({
        frame: frameData,
      }),
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to receive frame from server");
        }
        return response.json();
      })
      .then((data) => {
        const frameBase64 = data.frame;
        img.src = "data:image/jpeg;base64," + frameBase64;
        img.style.display = "inline"; // Show the image
        requestAnimationFrame(sendFrame);
      })
      .catch((error) => {
        console.error("Error receiving frame from server:", error);
        img.style.display = "none"; // Hide the image if there's an error
        requestAnimationFrame(sendFrame);
      });
  };
  sendFrame();
}

function stopWebcam() {
  sendFrames = false; // Set sendFrames flag to false to stop sending frames
  if (stream) {
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
  }
  video.srcObject = null;
  img.src = ""; // Clear the image source
  img.style.display = "none"; // Hide the image
}
