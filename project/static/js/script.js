document.addEventListener("DOMContentLoaded", function() {
    const uploadForm = document.querySelector(".upload-form");
    const loadingText = document.querySelector(".loading");

    uploadForm.addEventListener("submit", function() {
        loadingText.style.display = "block";
    });
});
