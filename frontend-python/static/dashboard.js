function navigateTo(url) {
    window.location.href = url;
}

function goToStudentPage() {
    const roll = document.getElementById("rollDropdown").value;
    if (roll) {
        window.location.href = "/student-behaviour/" + roll;
        return false;
    } else {
        alert("Please select a student.");
        return false;
    }
}
