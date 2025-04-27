document.addEventListener('DOMContentLoaded', function () {
    const questions = [
      "How many late night entries?",
      "Any weekend entries?",
      "How often absent during 9AM sessions?",
      "Average arrival time on weekdays?",
      "Any entries after 10 PM?",
      "Patterns of missing classes?",
      "Hostel late night activity?",
      "Is the student regular for afternoon classes?",
      "Longest gap between check-ins?",
      "Total number of late nights?"
    ];
  
    const questionsDiv = document.getElementById('questions');
    questions.forEach(q => {
      const btn = document.createElement('div');
      btn.className = 'question';
      btn.innerText = q;
      btn.addEventListener('click', () => {
        document.getElementById('query').value = q;
      });
      questionsDiv.appendChild(btn);
    });
  
    document.getElementById('askBtn').addEventListener('click', function () {
      const query = document.getElementById('query').value;
      if (!query.trim()) {
        alert('Please enter or select a question.');
        return;
      }
      fetch('/ask-groq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          roll_no: studentRollNo,
          query: query
        })
      })
      .then(res => res.json())
      .then(data => {
        document.getElementById('responseBox').innerText = data.reply || "No response.";
      })
      .catch(err => {
        document.getElementById('responseBox').innerText = "⚠️ Error contacting server.";
        console.error(err);
      });
    });
  });
  