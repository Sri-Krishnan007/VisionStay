document.addEventListener("DOMContentLoaded", function() {
    const editForms = document.querySelectorAll("form[action='/update']");

    editForms.forEach(form => {
        form.addEventListener("submit", function(e) {
            e.preventDefault(); // stop normal form submit

            const formData = new FormData(form);
            const jsonData = {};

            formData.forEach((value, key) => {
                if (key === 'is_hosteller' || key === 'is_blacklist') {
                    jsonData[key] = true; // checkbox checked
                } else {
                    jsonData[key] = value;
                }
            });

            // If checkbox missing, set false manually
            if (!formData.has('is_hosteller')) {
                jsonData['is_hosteller'] = false;
            }
            if (!formData.has('is_blacklist')) {
                jsonData['is_blacklist'] = false;
            }

            // 🛠 NEW LOGIC HERE: 
            if (jsonData['is_hosteller'] === false) {
                jsonData['room_no'] = ''; // Clear room number if not hosteller
            }

            console.log("Sending JSON:", jsonData); 

            fetch('/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(jsonData)
            })
            .then(response => {
                if (response.ok) {
                    alert("✅ Student updated successfully!");
                    location.reload(); // reload to refresh the list
                } else {
                    alert("❌ Update failed. Please try again.");
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert("❌ Update failed due to a network/server error.");
            });
        });
    });
});



function showTab(id) {
    const sections = document.querySelectorAll('.tab-section');
    sections.forEach(sec => sec.classList.remove('active-tab'));
    document.getElementById(id).classList.add('active-tab');
}

function toggleRoomField(checkbox, fieldId) {
    const roomInput = document.getElementById(fieldId);
    roomInput.style.display = checkbox.checked ? 'block' : 'none';
}

function validateAddForm() {
    const form = document.getElementById('addStudentForm');
    const fields = form.querySelectorAll('input[required]');
    for (let field of fields) {
        if (!field.value.trim()) {
            alert('Please fill out all required fields!');
            return false;
        }
    }
    return true;
}
