// Call showLoader when you start a task
function showLoader() {
    document.getElementById('loader').style.display = 'block';
}

// Call hideLoader when the task is complete
function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

function runTrain() {
    showLoader();
    const reelId = document.getElementById('train-input').value;
    localStorage.setItem('reelId', reelId);
    // Retrieve the reelId later with:
    // const reelId = localStorage.getItem('reelId');
    console.log(`Running training with Reel ID(s): ${reelId}`);
    
    // AJAX request to server to process the input
    fetch('/train/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            reelId: reelId
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.result);
    })
    .catch(error => {
        console.error('Error:', error);
    })
    .finally(() => {
        hideLoader();
    });
}

function runInference() {
    const searchText = document.getElementById('inference-input').value;
    localStorage.setItem('searchText', searchText);
    // Retrieve the searchText later with:
    // const searchText = localStorage.getItem('searchText');
    console.log(`Running inference with search text: ${searchText}`);
    
    // AJAX request to server to process the input
    fetch('/inference/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            searchText: searchText
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.result);

        const videoContainer = document.getElementById('video-container'); // You need to add this container in your HTML
        videoContainer.innerHTML = ''; // Clear any existing videos

        data.videoLinks.forEach(link => {
            const videoFrame = document.createElement('iframe');
            videoFrame.setAttribute('src', link);
            videoFrame.setAttribute('width', '560');
            videoFrame.setAttribute('height', '315');
            videoFrame.setAttribute('frameborder', '0');
            videoFrame.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture');
            videoFrame.setAttribute('allowfullscreen', true);
            videoContainer.appendChild(videoFrame);
        });
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
