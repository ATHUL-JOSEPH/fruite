/**
 * Fruit Freshness AI — Client-side Logic
 * Professional UI with smooth animations and robust handling.
 */

(function () {
    'use strict';

    // ── DOM References ──
    const uploadZone     = document.getElementById('upload-zone');
    const fileInput      = document.getElementById('file-input');
    const browseTrigger  = document.getElementById('browse-trigger');
    const previewStrip   = document.getElementById('preview-strip');
    const predictBtnWrap = document.getElementById('predict-btn-wrapper');
    const predictBtn     = document.getElementById('predict-btn');
    const resultsSection = document.getElementById('results-section');
    const resultsHeader  = document.getElementById('results-header');
    const resultsContainer = document.getElementById('results-container');
    const batchSummary   = document.getElementById('batch-summary');
    const batchTotal     = document.getElementById('batch-total');
    const batchFresh     = document.getElementById('batch-fresh');
    const batchRotten    = document.getElementById('batch-rotten');
    const toast          = document.getElementById('toast');
    const toastContent   = document.getElementById('toast-content');

    // ── State ──
    let selectedFiles = [];

    // ── Preloader ──
    window.addEventListener('load', () => {
        setTimeout(() => {
            const preloader = document.getElementById('preloader');
            if(preloader) preloader.classList.add('hidden');
        }, 600); // Ensures smooth transition after everything is ready
    });

    // ── Helpers ──
    function showToast(message, isError = false, duration = 3500) {
        const iconError = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        const iconWarning = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
        
        toastContent.innerHTML = `${isError ? iconError : iconWarning} ${message}`;
        toast.className = `toast ${isError ? 'error' : 'warning'} visible`;
        setTimeout(() => toast.classList.remove('visible'), duration);
    }

    function formatClassName(name) {
        const cleaned = name.replace(/fresh/i, 'Fresh ').replace(/rotten/i, 'Rotten ');
        return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
    }

    // ── Drag & Drop ──
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length === 0) {
            showToast('Please drop image files only', true);
            return;
        }
        handleFiles(files);
    });

    // ── Click to Browse ──
    uploadZone.addEventListener('click', () => fileInput.click());
    browseTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) handleFiles(files);
    });

    // ── Handle Selected Files ──
    function handleFiles(files) {
        selectedFiles = files;
        previewStrip.innerHTML = '';
        resultsSection.classList.remove('visible');
        resultsContainer.innerHTML = '';
        batchSummary.style.display = 'none';

        files.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-thumb';
                img.alt = file.name;
                img.style.animationDelay = `${index * 0.05}s`;
                previewStrip.appendChild(img);
            };
            reader.readAsDataURL(file);
        });

        predictBtnWrap.style.display = 'block';
        predictBtn.disabled = false;
        predictBtn.classList.remove('loading');
    }

    // ── Predict Button ──
    predictBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;

        predictBtn.disabled = true;
        predictBtn.classList.add('loading');

        try {
            if (selectedFiles.length === 1) {
                const formData = new FormData();
                formData.append('image', selectedFiles[0]);

                const res = await fetch('/predict', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(`Server error: ${res.status}`);
                const result = await res.json();
                if (result.error) throw new Error(result.error);

                const imgSrc = await readFileAsDataURL(selectedFiles[0]);
                showSingleResult(result, imgSrc);
            } else {
                const formData = new FormData();
                selectedFiles.forEach(f => formData.append('images', f));

                const res = await fetch('/predict-batch', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(`Server error: ${res.status}`);
                const results = await res.json();

                const imgSrcs = await Promise.all(selectedFiles.map(readFileAsDataURL));
                showBatchResults(results, imgSrcs);
            }
        } catch (err) {
            showToast(err.message, true);
            console.error(err);
        } finally {
            predictBtn.disabled = false;
            predictBtn.classList.remove('loading');
        }
    });

    function readFileAsDataURL(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    }

    // ── Show Single Result ──
    function showSingleResult(result, imgSrc) {
        resultsContainer.innerHTML = '';
        batchSummary.style.display = 'none';
        resultsHeader.textContent = 'Prediction Result';

        const card = createResultCard(result, imgSrc, 0);
        resultsContainer.appendChild(card);

        resultsSection.classList.add('visible');
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        requestAnimationFrame(() => {
            setTimeout(() => animateBars(card), 150);
            animateGauge(card, result.confidence, result.status === 'FRESH');
        });
    }

    // ── Show Batch Results ──
    function showBatchResults(results, imgSrcs) {
        resultsContainer.innerHTML = '';
        resultsHeader.textContent = `${results.length} Predictions`;

        const freshCount = results.filter(r => r.status === 'FRESH').length;
        const rottenCount = results.length - freshCount;
        batchTotal.textContent = results.length;
        batchFresh.textContent = freshCount;
        batchRotten.textContent = rottenCount;
        batchSummary.style.display = 'flex';

        results.forEach((result, i) => {
            const card = createResultCard(result, imgSrcs[i], i);
            resultsContainer.appendChild(card);
        });

        resultsSection.classList.add('visible');
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        const cards = resultsContainer.querySelectorAll('.result-card');
        cards.forEach((card, i) => {
            const result = results[i];
            setTimeout(() => {
                animateBars(card);
                animateGauge(card, result.confidence, result.status === 'FRESH');
            }, 200 + i * 100);
        });
    }

    // ── Create Result Card DOM ──
    function createResultCard(result, imgSrc, index) {
        const isFresh = result.status === 'FRESH';
        const statusClass = isFresh ? 'fresh' : 'rotten';
        
        const iconFresh = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        const iconRotten = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
        const statusIcon = isFresh ? iconFresh : iconRotten;
        
        const accentColor = isFresh ? 'var(--fresh-color)' : 'var(--rotten-color)';

        const card = document.createElement('div');
        card.className = 'glass-card-strong result-card';
        card.style.animationDelay = `${index * 0.1}s`;

        card.innerHTML = `
            <div class="result-image-wrapper ${statusClass}">
                <img class="result-image" src="${imgSrc}" alt="${result.fruit}">
            </div>
            <div class="result-details">
                <div class="result-header-row">
                    <div class="result-fruit-name">${result.fruit}</div>
                    <div class="status-badge ${statusClass}">
                        ${statusIcon} ${result.status}
                    </div>
                </div>

                <div class="confidence-section">
                    <div class="gauge-container">
                        <div class="gauge-ring" style="--gauge-color: ${accentColor}; --gauge-deg: 0deg;">
                            <div class="gauge-inner">
                                <span class="gauge-value">
                                    <span class="gauge-num">0</span><span class="percent">%</span>
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="confidence-label">
                        <strong>Confidence Score</strong>
                        <span>Model is ${result.confidence >= 90 ? 'highly' : result.confidence >= 70 ? 'moderately' : 'somewhat'} confident in this prediction.</span>
                    </div>
                </div>

                <div class="top5-section">
                    <div class="top5-title">Prediction Breakdown</div>
                    <div class="top5-list">
                        ${result.top5.map((item, i) => {
                            const barClass = item.is_fresh ? 'fresh-bar' : 'rotten-bar';
                            return `
                                <div class="top5-item">
                                    <span class="top5-name">${formatClassName(item.name)}</span>
                                    <div class="top5-bar-track">
                                        <div class="top5-bar-fill ${barClass}" data-width="${item.probability}"></div>
                                    </div>
                                    <span class="top5-value">${item.probability.toFixed(1)}%</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>
        `;

        return card;
    }

    // ── Animate Bars ──
    function animateBars(card) {
        const bars = card.querySelectorAll('.top5-bar-fill');
        bars.forEach((bar, index) => {
            const width = parseFloat(bar.getAttribute('data-width'));
            setTimeout(() => {
                bar.style.width = `${Math.max(width, 2)}%`;
            }, index * 100);
        });
    }

    // ── Animate Confidence Gauge ──
    function animateGauge(card, confidence, isFresh) {
        const gaugeRing = card.querySelector('.gauge-ring');
        const gaugeNum = card.querySelector('.gauge-num');
        const targetDeg = (confidence / 100) * 360;
        const duration = 1500;
        const startTime = performance.now();

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // easeOutExpo
            const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);

            const currentDeg = eased * targetDeg;
            const currentVal = eased * confidence;

            gaugeRing.style.setProperty('--gauge-deg', `${currentDeg}deg`);
            gaugeNum.textContent = Math.round(currentVal);

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        }

        requestAnimationFrame(step);
    }

})();
