const startButton = document.getElementById('startButton');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const resultsDiv = document.getElementById('results');
const statusDiv = document.getElementById('status');

let stream;
let intervalId = null;
const FPS = 5;

async function startCamera() {
  statusDiv.textContent = 'Requesting camera access...';
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      startButton.style.display = 'none';
      statusDiv.textContent = 'Camera active. Processing frames...';
      setTimeout(() => {
        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(captureAndSendFrame, 1000 / FPS);
      }, 500);
    };
    video.play();
  } catch (err) {
    console.error("Error accessing camera:", err);
    resultsDiv.textContent = `Camera Error: ${err.name} - ${err.message}`;
    statusDiv.textContent = 'Camera access failed.';
    startButton.style.display = 'block';
  }
}

async function captureAndSendFrame() {
  if (video.readyState < video.HAVE_ENOUGH_DATA) {
    return;
  }

  try {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    const response = await fetch('/detectface', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `HTTP error! Status: ${response.status}` }));
      console.error('API Error:', errorData);
      statusDiv.textContent = `API Error: ${errorData.error || response.statusText}`;
      resultsDiv.textContent = `Name: Error\nDistance: N/A\nTime: ${new Date().toLocaleTimeString()}\nLiveness: N/A\nMessage: ${errorData.message || 'Failed to get details'}`;
      return;
    }

    const data = await response.json();
    resultsDiv.textContent = `Name:     ${data.name}\nDistance: ${data.distance}\nTime:     ${data.time}\nLiveness: ${data.liveness}\n\nMessage:  ${data.message}`;
    statusDiv.textContent = 'Processing...';
  } catch (error) {
    console.error('Error capturing or sending frame:', error);
    statusDiv.textContent = `JavaScript Error: ${error.message}`;
  }
}

startButton.addEventListener('click', startCamera);

window.addEventListener('beforeunload', () => {
  if (intervalId) {
    clearInterval(intervalId);
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
});
