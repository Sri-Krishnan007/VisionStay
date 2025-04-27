document.addEventListener("DOMContentLoaded", function() {
    const adminLoginButton = document.getElementById('admin-login-button');
    const popupLoginButton = document.getElementById('popup-login-btn');
    const loginPopup = document.getElementById('login-popup');
    const loginForm = document.getElementById('login-form');

    // Manual Admin login button click
    adminLoginButton.addEventListener('click', function() {
        loginForm.classList.remove('d-none');
        loginPopup.classList.add('d-none');
    });

    // Popup button click
    popupLoginButton.addEventListener('click', function() {
        loginForm.classList.remove('d-none');
        loginPopup.classList.add('d-none');
    });

    // Auto popup after 10 seconds
    setTimeout(() => {
        loginPopup.classList.remove('d-none');
    }, 10000); // 10 seconds
});
