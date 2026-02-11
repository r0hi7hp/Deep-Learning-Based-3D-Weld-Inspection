/**
 * Weld Path Detection - Frontend JavaScript
 * Handles file uploads, API calls, and Plotly visualization
 * Enhanced with toast notifications and architecture display
 */

// State
const state = {
    cadFile: null,
    modelFile: null,
    isProcessing: false,
    architecture: null
};

// DOM Elements
const elements = {
    cadDropzone: document.getElementById('cad-dropzone'),
    cadInput: document.getElementById('cad-input'),
    cadFileInfo: document.getElementById('cad-file-info'),
    cadStatus: document.getElementById('cad-status'),
    cadCard: document.getElementById('cad-upload-card'),
    cadMeta: document.getElementById('cad-meta'),
    cadProgress: document.getElementById('cad-progress'),

    modelDropzone: document.getElementById('model-dropzone'),
    modelInput: document.getElementById('model-input'),
    modelFileInfo: document.getElementById('model-file-info'),
    modelStatus: document.getElementById('model-status'),
    modelCard: document.getElementById('model-upload-card'),
    archBadge: document.getElementById('arch-badge'),
    modelProgress: document.getElementById('model-progress'),

    processBtn: document.getElementById('process-btn'),
    plotlyContainer: document.getElementById('plotly-container'),
    vizPlaceholder: document.getElementById('viz-placeholder'),
    statsPanel: document.getElementById('stats-panel'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    loadingSubtext: document.getElementById('loading-subtext'),

    statArch: document.getElementById('stat-arch'),
    statTotal: document.getElementById('stat-total'),
    statWeld: document.getElementById('stat-weld'),
    statPercentage: document.getElementById('stat-percentage'),
    statConfidence: document.getElementById('stat-confidence'),

    toastContainer: document.getElementById('toast-container')
};

// File size limits (should match backend)
const MAX_CAD_SIZE_MB = 100;
const MAX_MODEL_SIZE_MB = 500;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropzone('cad');
    setupDropzone('model');
    setupProcessButton();
});

// Toast Notification System
function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type]}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;

    elements.toastContainer.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    // Auto-remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Setup Dropzone
function setupDropzone(type) {
    const dropzone = elements[`${type}Dropzone`];
    const input = elements[`${type}Input`];

    // Click to browse
    dropzone.addEventListener('click', (e) => {
        if (!e.target.classList.contains('remove-btn') &&
            !dropzone.querySelector('.file-info').style.display.includes('flex')) {
            input.click();
        }
    });

    // File input change
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(type, e.target.files[0]);
        }
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    // Drop
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(type, files[0]);
        }
    });
}

// Validate file size
function validateFileSize(file, type) {
    const sizeMB = file.size / (1024 * 1024);
    const maxSize = type === 'cad' ? MAX_CAD_SIZE_MB : MAX_MODEL_SIZE_MB;

    if (sizeMB > maxSize) {
        throw new Error(`File too large (${sizeMB.toFixed(1)}MB). Maximum is ${maxSize}MB.`);
    }

    if (file.size === 0) {
        throw new Error('File is empty.');
    }

    return sizeMB;
}

// Handle File Selection
async function handleFile(type, file) {
    const endpoint = type === 'cad' ? '/upload/cad' : '/upload/model';
    const fileInfo = elements[`${type}FileInfo`];
    const dropzoneContent = elements[`${type}Dropzone`].querySelector('.dropzone-content');
    const statusEl = elements[`${type}Status`];
    const card = elements[`${type}Card`];
    const progress = elements[`${type}Progress`];

    // Validate file type
    const validExtensions = type === 'cad' ? ['ply', 'stl'] : ['pth'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
        showToast(`Invalid file type. Please use ${validExtensions.join(' or ')} files.`, 'error');
        statusEl.textContent = `Invalid file type. Please use ${validExtensions.join(' or ')} files.`;
        statusEl.className = 'upload-status error';
        return;
    }

    // Validate file size
    try {
        validateFileSize(file, type);
    } catch (error) {
        showToast(error.message, 'error');
        statusEl.textContent = error.message;
        statusEl.className = 'upload-status error';
        return;
    }

    // Show uploading status
    statusEl.textContent = 'Uploading...';
    statusEl.className = 'upload-status';
    progress.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        progress.style.display = 'none';

        if (response.ok && result.success) {
            // Update state
            state[`${type}File`] = file;

            // Update UI
            dropzoneContent.style.display = 'none';
            fileInfo.style.display = 'flex';
            fileInfo.querySelector('.file-name').textContent = file.name;
            card.classList.add('uploaded');

            // Show additional info
            if (type === 'cad' && result.point_count) {
                elements.cadMeta.textContent = `${result.point_count.toLocaleString()} points`;
            }
            if (type === 'model' && result.architecture) {
                state.architecture = result.architecture;
                elements.archBadge.textContent = result.architecture;
                elements.archBadge.style.display = 'inline-block';
            }

            statusEl.textContent = result.message;
            statusEl.className = 'upload-status success';

            showToast(result.message, 'success');

            // Check if both files are uploaded
            updateProcessButton();
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        progress.style.display = 'none';
        showToast(error.message, 'error');
        statusEl.textContent = error.message;
        statusEl.className = 'upload-status error';
    }
}

// Remove File
function removeFile(type) {
    const input = elements[`${type}Input`];
    const fileInfo = elements[`${type}FileInfo`];
    const dropzoneContent = elements[`${type}Dropzone`].querySelector('.dropzone-content');
    const statusEl = elements[`${type}Status`];
    const card = elements[`${type}Card`];

    // Reset state
    state[`${type}File`] = null;
    input.value = '';

    // Update UI
    dropzoneContent.style.display = 'block';
    fileInfo.style.display = 'none';
    card.classList.remove('uploaded');
    statusEl.textContent = '';
    statusEl.className = 'upload-status';

    // Reset architecture badge if model
    if (type === 'model') {
        elements.archBadge.style.display = 'none';
        state.architecture = null;
    }

    // Reset meta if CAD
    if (type === 'cad') {
        elements.cadMeta.textContent = '';
    }

    updateProcessButton();
}

// Update Process Button State
function updateProcessButton() {
    elements.processBtn.disabled = !(state.cadFile && state.modelFile);
}

// Setup Process Button
function setupProcessButton() {
    elements.processBtn.addEventListener('click', processFiles);
}

// Process Files
async function processFiles() {
    if (state.isProcessing) return;

    state.isProcessing = true;
    elements.processBtn.classList.add('loading');
    elements.processBtn.disabled = true;
    elements.loadingOverlay.classList.add('active');

    // Update loading text
    elements.loadingText.textContent = 'Processing CAD Model...';
    elements.loadingSubtext.textContent = `Using ${state.architecture || 'Auto-detected'} architecture`;

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const result = await response.json();

        if (response.ok && result.success) {
            // Hide placeholder
            elements.vizPlaceholder.style.display = 'none';

            // Render Plotly chart
            renderPlotly(result.plotly);

            // Update stats
            updateStats(result.plotly.stats);

            showToast('Weld path detection complete!', 'success');
        } else {
            throw new Error(result.error || 'Processing failed');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    } finally {
        state.isProcessing = false;
        elements.processBtn.classList.remove('loading');
        elements.processBtn.disabled = false;
        elements.loadingOverlay.classList.remove('active');
    }
}

// Render Plotly Visualization
function renderPlotly(plotlyData) {
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['sendDataToCloud'],
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'weld_path_visualization',
            height: 1080,
            width: 1920,
            scale: 2
        }
    };

    // Clear existing plot
    Plotly.purge(elements.plotlyContainer);

    // Create new plot with animation
    Plotly.newPlot(
        elements.plotlyContainer,
        plotlyData.data,
        plotlyData.layout,
        config
    ).then(() => {
        // Add animation effect
        elements.plotlyContainer.style.opacity = '0';
        elements.plotlyContainer.style.transform = 'scale(0.95)';

        requestAnimationFrame(() => {
            elements.plotlyContainer.style.transition = 'all 0.5s ease';
            elements.plotlyContainer.style.opacity = '1';
            elements.plotlyContainer.style.transform = 'scale(1)';
        });
    });
}

// Update Statistics Panel
function updateStats(stats) {
    elements.statsPanel.style.display = 'flex';

    // Display architecture
    elements.statArch.textContent = stats.architecture || 'Unknown';

    // Animate numbers
    animateNumber(elements.statTotal, 0, stats.total_points, 1000);
    animateNumber(elements.statWeld, 0, stats.weld_points, 1000);

    elements.statPercentage.textContent = stats.weld_percentage.toFixed(2) + '%';
    elements.statConfidence.textContent = (stats.avg_confidence * 100).toFixed(1) + '%';
}

// Animate Number Counter
function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + range * easeProgress);

        element.textContent = current.toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Make removeFile available globally for onclick
window.removeFile = removeFile;
