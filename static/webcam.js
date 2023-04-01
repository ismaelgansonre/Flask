document.getElementById("start-webcam").addEventListener("click", function () {
    const video = document.getElementById("webcam");

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch(function (error) {
                console.log("Erreur lors de l'accès à la webcam : " + error);
            });
    } else {
        alert("Votre navigateur ne supporte pas l'accès à la webcam");
    }
});
