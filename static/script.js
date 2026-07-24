document.addEventListener('DOMContentLoaded', () => {
    const fileDropArea = document.getElementById('file-drop-area');
    const fileInput = document.getElementById('resume');
    const fileMsg = document.querySelector('.file-msg');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileMsg.textContent = fileInput.files[0].name;
        } else {
            fileMsg.textContent = 'or drag and drop here';
        }
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.add('is-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.remove('is-active'), false);
    });

    fileDropArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            fileMsg.textContent = files[0].name;
        }
    });

    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const dashboard = document.getElementById('dashboard');
    const errorBox = document.getElementById('error-box');
    const btn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const resumeFile = fileInput.files[0];
        const jdText = document.getElementById('jd_text').value;

        if (!resumeFile) {
            showError('Please upload a PDF resume.');
            return;
        }

        const formData = new FormData();
        formData.append('resume', resumeFile);
        formData.append('jd_text', jdText);

        errorBox.classList.add('hidden');
        dashboard.classList.add('hidden');
        dashboard.classList.remove('fade-in');
        loading.classList.remove('hidden');
        btn.disabled = true;

        try {
            const response = await fetch('/api/score', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                let msg = data.error || 'An unknown error occurred.';
                if (msg.includes('extract text')) {
                    msg = 'Could not extract text from this PDF. Please try a text-based PDF or OCR version.';
                }
                throw new Error(msg);
            }

            populateDashboard(data);

            loading.classList.add('hidden');
            dashboard.classList.remove('hidden');
            dashboard.classList.add('fade-in');
        } catch (err) {
            loading.classList.add('hidden');
            showError(err.message);
        } finally {
            btn.disabled = false;
        }
    });

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function renderEvidenceStrength(container, val) {
        if (!val) return;
        const badge = document.createElement('span');
        const cleanVal = val.toUpperCase();
        
        let label = 'Weak evidence';
        let cls = 'low';
        if (cleanVal === 'STRONG') {
            label = 'Strong evidence';
            cls = 'high';
        } else if (cleanVal === 'MODERATE') {
            label = 'Moderate evidence';
            cls = 'medium';
        }
        badge.className = `confidence-badge ${cls}`;
        badge.textContent = label;
        container.appendChild(badge);
    }

    function createTruncatedTextBlock(parent, text, limit = 220) {
        if (text.length <= limit) {
            parent.appendChild(document.createTextNode(`"... ${text} ..."`));
            return;
        }
        
        const container = document.createElement('span');
        container.className = 'snippet-text-container';
        
        const visiblePart = text.slice(0, limit);
        const hiddenPart = text.slice(limit);
        
        const visibleSpan = document.createElement('span');
        visibleSpan.textContent = `"... ${visiblePart}`;
        
        const hiddenSpan = document.createElement('span');
        hiddenSpan.textContent = `${hiddenPart}..."`;
        hiddenSpan.className = 'hidden';
        
        const btn = document.createElement('button');
        btn.className = 'snippet-toggle-btn';
        btn.textContent = 'Show More';
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            if (hiddenSpan.classList.contains('hidden')) {
                hiddenSpan.classList.remove('hidden');
                btn.textContent = 'Show Less';
            } else {
                hiddenSpan.classList.add('hidden');
                btn.textContent = 'Show More';
            }
        });
        
        container.appendChild(visibleSpan);
        container.appendChild(hiddenSpan);
        container.appendChild(btn);
        parent.appendChild(container);
    }

    function getQualityClass(label) {
        if (!label) return 'brief_mention';
        const lbl = label.toLowerCase();
        if (lbl.includes('strong')) return 'strong_implementation';
        if (lbl.includes('workflow')) return 'workflow_context';
        return 'brief_mention';
    }

    function populateDashboard(data) {
        // Fairness metrics removed from UI as requested

        // Input Quality Badge
        const iqBadge = document.getElementById('input-quality-badge');
        if (data.input_quality) {
            iqBadge.classList.remove('hidden');
            if (data.input_quality.status === 'OK') {
                iqBadge.textContent = 'Input Quality: Verified';
                iqBadge.className = 'badge pass';
            } else {
                iqBadge.textContent = 'Input Quality: Warnings Detected';
                iqBadge.className = 'badge fail';
            }
            
            const iqWarnBox = document.getElementById('input-quality-warning');
            if (data.input_quality.warnings && data.input_quality.warnings.length > 0) {
                renderMessageBox(iqWarnBox, 'Input Quality Warning:', data.input_quality.warnings.join(' '));
                iqWarnBox.classList.remove('hidden');
            } else {
                iqWarnBox.classList.add('hidden');
            }
        } else {
            iqBadge.classList.add('hidden');
        }

        // Fairness details panel removed from UI as requested


        const scoreCircle = document.getElementById('score-circle');
        const scoreValue = document.getElementById('score-value');
        const matchLabel = document.getElementById('match-label');
        const score = data.match_score ?? data.match_probability;
        const scorePct = Math.round(score * 100);

        scoreValue.textContent = `${scorePct}%`;
        matchLabel.textContent = data.match_label;

        scoreCircle.className = 'score-circle';
        if (scorePct >= 70) scoreCircle.classList.add('score-high');
        else if (scorePct >= 40) scoreCircle.classList.add('score-medium');
        else scoreCircle.classList.add('score-low');

        const vocabScore = data.extracted_features.tfidf_similarity;
        let vocabLabel = 'Weak';
        if (vocabScore > 0.35) vocabLabel = 'Strong';
        else if (vocabScore > 0.15) vocabLabel = 'Moderate';
        document.getElementById('feat-vocab').textContent = `${vocabLabel} Vocabulary Match (${vocabScore.toFixed(2)})`;

        const covScore = data.extracted_features.keyword_overlap_ratio;
        let covLabel = 'Weak';
        if (covScore > 0.40) covLabel = 'Strong';
        else if (covScore > 0.20) covLabel = 'Moderate';
        document.getElementById('feat-coverage').textContent = `${covLabel} Requirement Coverage (${(covScore * 100).toFixed(0)}%)`;

        const eduStatus = data.education_status;
        let eduText = '✓ Meets Job Requirement';
        if (eduStatus) {
            if (eduStatus.label === 'EXCEEDS') eduText = '✓ Exceeds Job Requirement';
            else if (eduStatus.label === 'BELOW') eduText = '⚠ Below Job Requirement';
        } else {
            const eduScore = data.extracted_features.education_level_score;
            if (eduScore > 0) eduText = '✓ Exceeds Job Requirement';
            else if (eduScore < 0) eduText = '⚠ Below Job Requirement';
        }
        
        let degreeFound = data.extracted_features.extracted_education_level || "unknown";
        degreeFound = degreeFound.charAt(0).toUpperCase() + degreeFound.slice(1);
        if (degreeFound !== 'Unknown') degreeFound += " degree detected.";
        else degreeFound = "Degree level not clearly detected.";
        
        const eduNode = document.getElementById('feat-edu');
        eduNode.innerHTML = `<strong>${eduText}</strong><br><span style="font-size:0.85em; color:var(--text-secondary)">${degreeFound}</span>`;

        const expGap = data.extracted_features.experience_gap;
        const expType = data.extracted_features.experience_type || {};
        const jdRequiresExp = data.extracted_features.jd_requires_experience !== false;

        const expFeatureLabel = document.getElementById('exp-feature-label');
        const expFeatureSubtext = document.getElementById('exp-feature-subtext');
        const expNode = document.getElementById('feat-exp');
        
        if (jdRequiresExp) {
            expFeatureLabel.textContent = 'Experience vs Requirement';
            expFeatureSubtext.textContent = 'Candidate yrs - JD requirement';
            
            let expText = '';
            if (expGap > 0) {
                expText = `✓ Exceeds (+${expGap.toFixed(1)} yrs)`;
            } else if (expGap < 0) {
                expText = `⚠ Gap (${expGap.toFixed(1)} yrs)`;
            } else {
                if (data.extracted_features.extracted_years > 0) {
                    expText = '✓ Matches requirement (0 yrs gap)';
                } else {
                    if (expType.professional) {
                        expText = '✓ Professional experience detected (0 yrs parsed)';
                    } else if (expType.internship || expType.freelance || expType.project) {
                        expText = '⚠ Professional experience not detected. ';
                        let found = [];
                        if (expType.internship) found.push('Internships');
                        if (expType.freelance) found.push('Freelance');
                        if (expType.project) found.push('Projects');
                        expText += found.join(', ') + ' detected.';
                    } else {
                        expText = '⚠ No experience or projects detected.';
                    }
                }
            }
            expNode.textContent = expText;
            renderEvidenceStrength(expNode, data.extracted_features.experience_confidence || data.extracted_features.experience_evidence_strength);
        } else {
            expFeatureLabel.textContent = 'Professional Experience';
            expFeatureSubtext.textContent = 'The Job Description does not explicitly require prior professional experience.';
            
            let expText = '';
            if (data.extracted_features.extracted_years > 0 || expType.professional) {
                let yrs = data.extracted_features.extracted_years > 0 ? ` (${data.extracted_features.extracted_years.toFixed(1)} yrs)` : '';
                expText = `✓ Professional work experience detected${yrs}.`;
            } else if (expType.internship || expType.freelance || expType.project) {
                let found = [];
                if (expType.internship) found.push('Internships');
                if (expType.freelance) found.push('Freelance');
                if (expType.project) found.push('Projects');
                expText = `Professional work experience not detected.<br><br>✓ ${found.join(', ')} detected.`;
            } else {
                expText = 'No professional experience or projects detected.';
            }
            expNode.innerHTML = expText;
        }

        const warnBox = document.getElementById('divergence-warning');
        if (data.divergence_warning) {
            renderMessageBox(warnBox, 'Divergence Warning:', data.divergence_warning);
            warnBox.classList.remove('hidden');
        } else {
            warnBox.classList.add('hidden');
        }

        // Render structured recommendations (Severity Tiers & Recruiter/Candidate split)
        const recBox = document.getElementById('recommendation-box');
        recBox.textContent = '';
        if ((data.recruiter_notes && data.recruiter_notes.length > 0) || (data.candidate_advice && data.candidate_advice.length > 0)) {
            // Recruiter Notes
            if (data.recruiter_notes && data.recruiter_notes.length > 0) {
                const recHeader = document.createElement('strong');
                recHeader.textContent = 'Recruiter Human-Review Risk Notes:';
                recBox.appendChild(recHeader);
                
                const recList = document.createElement('div');
                recList.style.marginTop = '0.5rem';
                recList.style.marginBottom = '1rem';
                
                data.recruiter_notes.forEach(note => {
                    const noteCard = document.createElement('div');
                    const tier = note.tier.toLowerCase();
                    noteCard.className = `recruiter-note-card ${tier}`;
                    
                    const tag = document.createElement('span');
                    tag.className = `tier-tag ${tier}`;
                    tag.textContent = note.tier;
                    
                    noteCard.appendChild(tag);
                    noteCard.appendChild(document.createTextNode(note.message));
                    recList.appendChild(noteCard);
                });
                recBox.appendChild(recList);
            }

            // Candidate Advice
            if (data.candidate_advice && data.candidate_advice.length > 0) {
                const candHeader = document.createElement('strong');
                candHeader.textContent = 'Candidate Visibility & Alignment Advice:';
                recBox.appendChild(candHeader);
                
                const candList = document.createElement('ul');
                candList.className = 'recommendation-list';
                candList.style.marginBottom = '0.5rem';
                
                data.candidate_advice.forEach(advice => {
                    const li = document.createElement('li');
                    li.textContent = advice;
                    candList.appendChild(li);
                });
                recBox.appendChild(candList);
            }
            recBox.classList.remove('hidden');
        } else {
            recBox.classList.add('hidden');
        }

        // Render Matched Evidence Snippets dynamically (Top 5 scroll-contained list)
        let evidenceBox = document.getElementById('evidence-box');
        if (!evidenceBox) {
            evidenceBox = document.createElement('div');
            evidenceBox.id = 'evidence-box';
            evidenceBox.className = 'evidence-section';
            const limBox = document.querySelector('.limitations-box');
            if (limBox) {
                limBox.parentNode.insertBefore(evidenceBox, limBox);
            }
        }
        evidenceBox.textContent = '';
        if (data.evidence) {
            const header = document.createElement('h3');
            header.textContent = 'Matched Technical Skills Evidence (Top 5)';
            evidenceBox.appendChild(header);

            if (data.evidence.matched && data.evidence.matched.length > 0) {
                const list = document.createElement('ul');
                list.className = 'snippet-list';
                
                data.evidence.matched.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'snippet-item';
                    
                    const p = document.createElement('p');
                    createTruncatedTextBlock(p, item.snippet, 220);
                    li.appendChild(p);

                    const span = document.createElement('span');
                    span.className = 'terms';
                    span.textContent = `Matched Terms: ${item.terms.join(', ')}`;
                    li.appendChild(span);

                    // Evidence quality qualifier tag badge is removed to reduce repetitive UI text
                    // Display only the technical skill and the supporting excerpt as requested
                    list.appendChild(li);
                });
                evidenceBox.appendChild(list);
            } else {
                const p = document.createElement('p');
                p.style.fontSize = '0.85rem';
                p.style.color = 'var(--text-secondary)';
                p.textContent = 'No specific technical skills found in context.';
                evidenceBox.appendChild(p);
            }

            if (data.evidence.missing && data.evidence.missing.length >= 3) {
                const subh = document.createElement('h3');
                subh.style.marginTop = '1.5rem';
                subh.textContent = 'Potentially Missing Technical Skills';
                evidenceBox.appendChild(subh);

                const listDiv = document.createElement('div');
                listDiv.className = 'missing-keywords-list';

                data.evidence.missing.forEach(term => {
                    const tag = document.createElement('span');
                    tag.className = 'missing-kw-tag';
                    tag.textContent = term;
                    listDiv.appendChild(tag);
                });
                evidenceBox.appendChild(listDiv);
            }
            evidenceBox.classList.remove('hidden');
        } else {
            evidenceBox.classList.add('hidden');
        }

        // Render soft conversational SHAP insights
        let shapInsightsBox = document.getElementById('shap-insights-box');
        if (!shapInsightsBox) {
            shapInsightsBox = document.createElement('div');
            shapInsightsBox.id = 'shap-insights-box';
            shapInsightsBox.className = 'shap-insights-container';
            shapInsightsBox.style.marginTop = '1rem';
            const shapSec = document.querySelector('.shap-section');
            if (shapSec) {
                shapSec.appendChild(shapInsightsBox);
            }
        }
        shapInsightsBox.textContent = '';
        if (data.shap_insights) {
            const list = document.createElement('ul');
            list.style.listStyle = 'none';
            list.style.padding = '0';
            list.style.margin = '0';

            const featureLabels = {
                tfidf_similarity: 'Vocabulary Match',
                keyword_overlap_ratio: 'Requirement Coverage',
                education_level_score: 'Education Fit',
                experience_gap: 'Experience Gap'
            };

            for (const [feat, insight] of Object.entries(data.shap_insights)) {
                if (insight.impact === 'neutral') continue;

                const li = document.createElement('li');
                li.className = 'shap-insight-text';
                
                const strong = document.createElement('strong');
                strong.textContent = `${featureLabels[feat] || feat}: `;
                li.appendChild(strong);
                li.appendChild(document.createTextNode(insight.summary));
                list.appendChild(li);
            }
            shapInsightsBox.appendChild(list);
            shapInsightsBox.classList.remove('hidden');
        } else {
            shapInsightsBox.classList.add('hidden');
        }

        renderShapChart(data.shap_values);
    }

    function renderMessageBox(container, title, message) {
        container.textContent = '';
        const strong = document.createElement('strong');
        strong.textContent = title + ' ';
        container.appendChild(strong);
        container.appendChild(document.createTextNode(message));
    }

    function renderShapChart(shapData) {
        const container = document.getElementById('shap-bar-chart');
        if (!container) return;
        
        container.textContent = '';
        
        document.getElementById('shap-base-text').textContent =
            `Base Value: ${shapData.base_value} | Final Log Odds: ${shapData.final_log_odds}`;
            
        const labels = {
            tfidf_similarity: 'Vocabulary Match',
            keyword_overlap_ratio: 'Requirement Coverage',
            education_level_score: 'Education Fit',
            experience_gap: 'Experience Gap'
        };
        
        const wrapper = document.createElement('div');
        wrapper.className = 'shap-chart-wrapper';
        
        const values = [
            shapData.tfidf_similarity,
            shapData.keyword_overlap_ratio,
            shapData.education_level_score,
            shapData.experience_gap
        ];
        const maxAbs = Math.max(...values.map(Math.abs), 1.0);
        
        for (const [feature, score] of Object.entries(labels)) {
            const val = shapData[feature] || 0.0;
            
            const row = document.createElement('div');
            row.className = 'shap-bar-row';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'shap-bar-label';
            labelDiv.textContent = score;
            
            const axisDiv = document.createElement('div');
            axisDiv.className = 'shap-bar-axis';
            
            const midline = document.createElement('div');
            midline.className = 'shap-bar-midline';
            axisDiv.appendChild(midline);
            
            const fill = document.createElement('div');
            const pct = Math.min((Math.abs(val) / maxAbs) * 50, 50);
            
            if (val >= 0) {
                fill.className = 'shap-bar-fill positive';
                fill.style.width = `${pct}%`;
                fill.style.left = '50%';
            } else {
                fill.className = 'shap-bar-fill negative';
                fill.style.width = `${pct}%`;
                fill.style.left = `${50 - pct}%`;
            }
            axisDiv.appendChild(fill);
            
            const valDiv = document.createElement('div');
            valDiv.className = 'shap-bar-value';
            valDiv.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)}`;
            
            row.appendChild(labelDiv);
            row.appendChild(axisDiv);
            row.appendChild(valDiv);
            wrapper.appendChild(row);
        }
        container.appendChild(wrapper);
    }
});
