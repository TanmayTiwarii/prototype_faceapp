document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const tools = document.querySelectorAll('.tool-card:not(.disabled)');
    const resultSection = document.getElementById('result-section');
    const loader = document.getElementById('loader');
    const resultText = document.getElementById('result-text');

    let currentFile = null;

    // Handle file upload
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            currentFile = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.classList.remove('hidden');
                resultSection.classList.add('hidden');
            };
            
            reader.readAsDataURL(currentFile);
        }
    });

    // Execute feature function
    window.executeFeature = async (tool) => {
        if (!currentFile) {
            alert('Please upload a photo first!');
            return;
        }

            const feature = tool.getAttribute('data-feature');
            
            // Show processing state
            resultSection.classList.remove('hidden');
            loader.classList.remove('hidden');
            resultText.innerText = `Processing with ${tool.querySelector('h3').innerText}...`;
            
            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth' });

            try {
                // Prepare form data
                const formData = new FormData();
                formData.append('file', currentFile);
                if (feature === 'hair-color') {
                    const colorPicker = document.getElementById('hair-color-picker');
                    if (colorPicker) {
                        formData.append('color', colorPicker.value);
                    }
                }

                // Call FastAPI backend
                const response = await fetch(`/api/${feature}`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // Hide loader and show result
                loader.classList.add('hidden');
                
                if (data.status === 'error') {
                    resultText.innerHTML = `<span style="color: #ff4444;">Error: ${data.message}</span>`;
                    return;
                }

                // If backend returns a processed image, update the preview
                if (data.result_b64) {
                    previewImg.src = data.result_b64;
                }
                
                if (feature === 'face-shape') {
                    resultText.innerHTML = `<strong>Success!</strong> ${data.message} <br> Detected Shape: <span style="color: #ff007f">${data.shape}</span>`;
                } else {
                    resultText.innerHTML = `<strong>Success!</strong> ${data.message}`;
                }
                
        } catch (error) {
            console.error('Error:', error);
            loader.classList.add('hidden');
            resultText.innerHTML = `<span style="color: #ff4444;">Error processing image. Make sure the FastAPI backend is running!</span>`;
        }
    };

    // Bind event listeners
    tools.forEach(tool => {
        if (tool.getAttribute('data-feature') === 'hair-color') {
            // Do not bind click to the whole card for hair-color
            return;
        }
        tool.addEventListener('click', () => executeFeature(tool));
    });

    // Explicit trigger for Hair Color OK button
    window.triggerHairColor = () => {
        const tool = document.querySelector('.tool-card[data-feature="hair-color"]');
        executeFeature(tool);
    };
});
