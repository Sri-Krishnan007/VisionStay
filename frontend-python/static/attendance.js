let mode = '';

function switchMode(selectedMode) {
  mode = selectedMode;

  document.getElementById('filterForm').classList.remove('d-none');
  document.getElementById('overallTable').classList.add('d-none');
  document.getElementById('particularTable').classList.add('d-none');

  if (mode === 'overall') {
    document.getElementById('studentSelectContainer').style.display = 'none';
  } else {
    document.getElementById('studentSelectContainer').style.display = 'block';
  }
}

function loadAttendance() {
  const start = document.getElementById('startDate').value;
  const end = document.getElementById('endDate').value;
  const studentId = document.getElementById('studentSelect').value;

  if (!start || !end) {
    alert('⚠️ Please select both Start and End dates!');
    return;
  }

  if (mode === 'overall') {
    renderOverallAttendance(start, end);
  } else if (mode === 'particular') {
    if (!studentId) {
      alert('⚠️ Please select a student!');
      return;
    }
    renderParticularAttendance(start, end, studentId);
  }
}

function renderParticularAttendance(start, end, studentId) {
    let student = studentsData.find(s => s.roll_no == studentId);
    if (!student || !student.attendance) {
      document.getElementById('particularTable').innerHTML = `<p class="text-center text-danger">No data found!</p>`;
      return;
    }
  
    const startDate = new Date(start);
    const endDate = new Date(end);
  
    const filteredLogs = student.attendance.filter(log => {
      const logDate = new Date(log.timestamp);
      return logDate >= startDate && logDate <= endDate;
    });
  
    let table = `
      <table class="table table-bordered table-hover">
        <thead class="table-success text-center">
          <tr><th>Status</th><th>Time</th></tr>
        </thead>
        <tbody>`;
  
    filteredLogs.forEach(log => {
      const timeFormatted = new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      table += `<tr>
        <td><span class="badge badge-${log.status}">${log.status}</span></td>
        <td>${timeFormatted}</td>
      </tr>`;
    });
  
    table += `</tbody></table>`;
  
    document.getElementById('particularTable').innerHTML = table;
    document.getElementById('particularTable').classList.remove('d-none');
  }
  

function renderOverallAttendance(start, end) {
    const startDate = new Date(start);
    const endDate = new Date(end);
  
    // Extract month and year
    const monthName = startDate.toLocaleString('default', { month: 'long' });
    const year = startDate.getFullYear();
  
    let table = `
      <h4 class="text-center text-secondary mb-3">${monthName} ${year}</h4> 
      <table class="table table-bordered table-hover">
        <thead class="table-primary text-center">
          <tr><th>Student Name</th>`;
  
    // Only loop between selected start and end day
    const startDay = startDate.getDate();
    const endDay = endDate.getDate();
  
    for (let day = startDay; day <= endDay; day++) {
      table += `<th>${day}</th>`;
    }
  
    table += `</tr></thead><tbody>`;
  
    studentsData.forEach(student => {
      table += `<tr><td>${student.name}</td>`;
  
      let dayStatusMap = {};
  
      student.attendance?.forEach(log => {
        const logDate = new Date(log.timestamp);
        const logDay = logDate.getDate();
        const logMonth = logDate.getMonth();
        const logYear = logDate.getFullYear();
  
        // Make sure month and year also match, not just day
        if (logMonth === startDate.getMonth() && logYear === startDate.getFullYear()) {
          if (!dayStatusMap[logDay] || log.status === 'IN') {
            dayStatusMap[logDay] = log.status;
          }
        }
      });
  
      for (let day = startDay; day <= endDay; day++) {
        const status = dayStatusMap[day] || 'NIL';
        table += `<td><span class="badge badge-${status}">${status}</span></td>`;
      }
  
      table += `</tr>`;
    });
  
    table += `</tbody></table>`;
  
    document.getElementById('overallTable').innerHTML = table;
    document.getElementById('overallTable').classList.remove('d-none');
  }
  