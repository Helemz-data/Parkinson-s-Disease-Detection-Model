const form = document.getElementById("predictionForm");
const loading = document.getElementById("loading");
const resultDiv = document.getElementById("result");
const voiceToggle = document.getElementById("voiceToggle");
const voiceSection = document.getElementById("voiceSection");

voiceToggle.addEventListener("change", () => {
    voiceSection.style.display = voiceToggle.checked ? "block" : "none";
});

form.addEventListener("submit", function (e) {
    e.preventDefault();

    const inputs = document.querySelectorAll(".feature-input");
    const formData = new FormData();

    for (let input of inputs) {
        if (input.value === "") {
            alert("All fields are required.");
            return;
        }
        formData.append(input.name, input.value);
    }

    const modelChoice = document.getElementById("modelSelect").value;
    formData.append("model", modelChoice);

    const voiceFile = document.getElementById("voiceFile").files[0];
    if (voiceFile) {
        formData.append("voice", voiceFile);
    }

    loading.style.display = "block";
    resultDiv.style.display = "none";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loading.style.display = "none";
        resultDiv.style.display = "block";

        resultDiv.innerText =
            data.prediction === 1
                ? "Parkinson’s Disease Detected"
                : "No Parkinson’s Disease Detected";

        const confidence = (data.probability * 100).toFixed(1);
        document.getElementById("confidenceBar").style.display = "block";
        document.getElementById("confidenceFill").style.width = confidence + "%";
        document.getElementById("confidenceFill").innerText = confidence + "%";
    })
    .catch(err => {
        loading.style.display = "none";
        alert("Prediction failed. Check console.");
        console.error(err);
    });
});


