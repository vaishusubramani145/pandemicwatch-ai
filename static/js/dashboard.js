/**
 * PandemicWatch AI - Frontend Dashboard Controller
 * Handles Chart.js charts, Leaflet geospatial visualization, SEIR simulation sliders,
 * REST API communication, and automated situation report formatting.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        region: document.getElementById('regionSelect').value,
        pathogen: document.getElementById('pathogenSelect').value,
        horizon: parseInt(document.getElementById('horizonSelect').value, 10),
        map: null,
        mapMarkers: [],
        charts: {
            forecast: null,
            signals: null,
            seir: null
        }
    };

    // Initialize Map
    function initMap() {
        if (state.map) return;
        state.map = L.map('mapContainer', {
            zoomControl: true,
            scrollWheelZoom: false
        }).setView([20.5937, 78.9629], 3);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(state.map);
    }

    // Chart Defaults
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // Initialize Forecast Chart
    function initForecastChart() {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        state.charts.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Historical Cases',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2.2,
                        pointRadius: 1.5,
                        tension: 0.3,
                        fill: false
                    },
                    {
                        label: 'ML Predicted Cases',
                        data: [],
                        borderColor: '#ef4444',
                        borderWidth: 2.4,
                        borderDash: [5, 4],
                        pointRadius: 3,
                        pointBackgroundColor: '#ef4444',
                        tension: 0.3,
                        fill: false
                    },
                    {
                        label: 'Upper 95% CI',
                        data: [],
                        borderColor: 'transparent',
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Lower 95% CI',
                        data: [],
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(239, 68, 68, 0.14)',
                        pointRadius: 0,
                        fill: '-1' // Fill up to Upper CI
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#111827',
                        borderColor: '#374151',
                        borderWidth: 1,
                        padding: 10,
                        titleFont: { weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { maxTicksLimit: 12, font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: {
                            callback: (v) => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v,
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    // Initialize Signals Chart (Rt, TPR, Mobility)
    function initSignalsChart() {
        const ctx = document.getElementById('signalsChart').getContext('2d');
        state.charts.signals = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Effective Rt',
                        data: [],
                        borderColor: '#10b981',
                        borderWidth: 2.2,
                        pointRadius: 0,
                        yAxisID: 'yRt',
                        tension: 0.3
                    },
                    {
                        label: 'Test Positivity Rate (TPR %)',
                        data: [],
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'yPct',
                        tension: 0.3
                    },
                    {
                        label: 'Mobility Index %',
                        data: [],
                        borderColor: '#8b5cf6',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        yAxisID: 'yPct',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, padding: 12, font: { size: 11 } }
                    },
                    tooltip: { backgroundColor: '#111827', borderColor: '#374151', borderWidth: 1 }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { maxTicksLimit: 10, font: { size: 11 } }
                    },
                    yRt: {
                        position: 'left',
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Rt Value', color: '#10b981', font: { size: 11 } },
                        min: 0.4,
                        max: 3.5
                    },
                    yPct: {
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        title: { display: true, text: 'Percentage (%)', color: '#f59e0b', font: { size: 11 } },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    // Initialize SEIR Simulation Chart
    function initSeirChart() {
        const ctx = document.getElementById('seirChart').getContext('2d');
        state.charts.seir = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Unmitigated Baseline (Infected)',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'With Policy Interventions (Infected)',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.18)',
                        borderWidth: 2.4,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Acute Hospital Bed Need',
                        data: [],
                        borderColor: '#f59e0b',
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: false,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, padding: 10, font: { size: 11 } }
                    },
                    tooltip: { backgroundColor: '#111827', borderColor: '#374151', borderWidth: 1 }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        title: { display: true, text: 'Simulation Timeline (Days)', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: {
                            callback: (v) => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v,
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    // Load Overview Data
    async function loadOverview() {
        try {
            const res = await fetch(`/api/overview?region=${encodeURIComponent(state.region)}&pathogen=${encodeURIComponent(state.pathogen)}`);
            const data = await res.json();

            // Update Metadata & Status
            document.getElementById('lastUpdatedTime').textContent = new Date().toLocaleTimeString();
            document.getElementById('transmissionTypeTag').innerHTML = `<i class="fa-solid fa-dna"></i> ${data.metadata.transmission_type}`;
            
            const anomalyPill = document.getElementById('anomalyPill');
            if (data.kpis.recent_anomaly_detected) {
                anomalyPill.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-danger"></i> Anomaly Alert: Surveillance Spike Detected`;
                anomalyPill.style.borderColor = '#ef4444';
            } else {
                anomalyPill.innerHTML = `<i class="fa-solid fa-shield-check text-success"></i> Anomaly Guard: Nominal`;
                anomalyPill.style.borderColor = '';
            }

            // Update KPIs
            const ewi = data.kpis.ewi;
            const ewiCard = document.getElementById('ewiCard');
            const ewiVal = document.getElementById('ewiValue');
            const ewiBadge = document.getElementById('ewiBadge');
            
            ewiVal.textContent = ewi.score;
            ewiVal.style.color = ewi.color;
            ewiBadge.textContent = ewi.tier;
            ewiBadge.style.background = ewi.color + '25';
            ewiBadge.style.color = ewi.color;
            ewiCard.style.borderLeftColor = ewi.color;
            document.getElementById('ewiAction').textContent = ewi.recommended_action;

            // Rt
            const rtVal = document.getElementById('rtValue');
            const rtBadge = document.getElementById('rtBadge');
            rtVal.textContent = data.kpis.current_rt;
            if (data.kpis.current_rt >= 1.0) {
                rtVal.style.color = '#ef4444';
                rtBadge.textContent = 'Expanding (Rt > 1)';
                rtBadge.style.background = 'rgba(239, 68, 68, 0.2)';
                rtBadge.style.color = '#ef4444';
            } else {
                rtVal.style.color = '#10b981';
                rtBadge.textContent = 'Contained (Rt < 1)';
                rtBadge.style.background = 'rgba(16, 185, 129, 0.2)';
                rtBadge.style.color = '#10b981';
            }

            // 7-day Velocity
            document.getElementById('cases7dValue').textContent = data.kpis.seven_day_avg_cases.toLocaleString();
            document.getElementById('dailyCasesValue').textContent = data.kpis.latest_daily_cases.toLocaleString();
            const growthDelta = document.getElementById('growthDelta');
            const gVal = data.kpis.weekly_growth_pct;
            growthDelta.textContent = (gVal >= 0 ? '+' : '') + gVal + '%';
            growthDelta.className = 'kpi-delta ' + (gVal > 0 ? 'text-danger' : 'text-success');

            // TPR & Vax
            document.getElementById('tprValue').textContent = data.kpis.latest_tpr_pct + '%';
            document.getElementById('vaxValue').textContent = data.kpis.vaccination_pct + '%';

            // ICU
            const icuOccupancy = data.kpis.icu_occupancy_pct;
            document.getElementById('icuPctValue').textContent = icuOccupancy + '%';
            const icuBadge = document.getElementById('icuStatusBadge');
            icuBadge.textContent = data.kpis.exhaustion.risk_level;
            if (data.kpis.exhaustion.risk_level === 'CRITICAL' || data.kpis.exhaustion.risk_level === 'SEVERE') {
                icuBadge.className = 'kpi-status-badge text-danger';
            } else if (data.kpis.exhaustion.risk_level === 'ELEVATED') {
                icuBadge.className = 'kpi-status-badge text-warning';
            } else {
                icuBadge.className = 'kpi-status-badge text-success';
            }
            document.getElementById('daysToExhaustion').textContent = data.kpis.exhaustion.days_to_exhaustion;

            // Signals Chart Update
            const ts = data.timeseries;
            state.charts.signals.data.labels = ts.dates;
            state.charts.signals.data.datasets[0].data = ts.rt;
            state.charts.signals.data.datasets[1].data = ts.tpr;
            state.charts.signals.data.datasets[2].data = ts.mobility;
            state.charts.signals.update();

            // Populate Regional Ranking Table & Map Markers
            renderRegionalTableAndMap(data.all_regions_summary);

            // Load ML Forecast
            await loadForecast(ts);

        } catch (err) {
            console.error('Error loading overview:', err);
        }
    }

    // Load ML Forecast
    async function loadForecast(timeseries) {
        try {
            document.getElementById('statHorizon').textContent = state.horizon + ' Days';
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    region: state.region,
                    pathogen: state.pathogen,
                    horizon: state.horizon
                })
            });
            const predData = await res.json();

            // Stitch historical cases and future predictions
            const histDates = timeseries.dates;
            const histCases = timeseries.daily_cases;
            const futureDates = predData.dates;

            const allLabels = [...histDates, ...futureDates];

            // Historical dataset (padded with nulls for future)
            const histDataPadded = [...histCases, ...Array(futureDates.length).fill(null)];

            // Forecast dataset (starts from the last historical point to connect lines)
            const lastHistCase = histCases[histCases.length - 1];
            const predDataPadded = [
                ...Array(histDates.length - 1).fill(null),
                lastHistCase,
                ...predData.predicted_cases
            ];

            const upperCiPadded = [
                ...Array(histDates.length - 1).fill(null),
                lastHistCase,
                ...predData.upper_95_ci
            ];

            const lowerCiPadded = [
                ...Array(histDates.length - 1).fill(null),
                lastHistCase,
                ...predData.lower_95_ci
            ];

            state.charts.forecast.data.labels = allLabels;
            state.charts.forecast.data.datasets[0].data = histDataPadded;
            state.charts.forecast.data.datasets[1].data = predDataPadded;
            state.charts.forecast.data.datasets[2].data = upperCiPadded;
            state.charts.forecast.data.datasets[3].data = lowerCiPadded;
            state.charts.forecast.update();

        } catch (err) {
            console.error('Error loading forecast:', err);
        }
    }

    // Run SEIR Simulation
    async function runSimulation() {
        const distancing = parseFloat(document.getElementById('distRange').value);
        const masks = parseFloat(document.getElementById('maskRange').value);
        const lockdown = parseFloat(document.getElementById('lockdownRange').value);
        const vaxRate = parseFloat(document.getElementById('vaxRateRange').value) / 100.0;
        const startDay = parseInt(document.getElementById('startDayRange').value, 10);
        const r0 = parseFloat(document.getElementById('r0Range').value) / 10.0;

        try {
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    distancing: distancing,
                    masks: masks,
                    lockdown: lockdown,
                    vax_rate: vaxRate,
                    intervention_start: startDay,
                    r0: r0,
                    population: 10_000_000,
                    days: 120
                })
            });
            const sim = await res.json();

            // Update impact metrics
            document.getElementById('deathsAvertedVal').textContent = sim.impact.deaths_averted.toLocaleString();
            document.getElementById('peakReductionVal').textContent = sim.impact.peak_reduction_pct + '%';
            document.getElementById('peakDelayVal').textContent = '+' + sim.impact.peak_day_shift + ' Days';
            document.getElementById('intervenedRtVal').textContent = sim.impact.final_rt.toFixed(2);

            // Update SEIR Chart
            state.charts.seir.data.labels = sim.baseline.days.map(d => `Day ${d}`);
            state.charts.seir.data.datasets[0].data = sim.baseline.infected;
            state.charts.seir.data.datasets[1].data = sim.intervened.infected;
            state.charts.seir.data.datasets[2].data = sim.intervened.hospitalized;
            state.charts.seir.update();

        } catch (err) {
            console.error('Error running simulation:', err);
        }
    }

    // Render Regional Table and Leaflet Map
    function renderRegionalTableAndMap(regionsList) {
        const tbody = document.getElementById('regionalTableBody');
        tbody.innerHTML = '';

        // Clear existing map markers
        state.mapMarkers.forEach(m => state.map.removeLayer(m));
        state.mapMarkers = [];

        regionsList.forEach(reg => {
            // Table row
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${reg.region}</strong></td>
                <td><span class="badge-tier" style="background:${reg.color}25; color:${reg.color}">${reg.tier}</span></td>
                <td><strong style="color:${reg.color}">${reg.ewi_score}</strong></td>
                <td>${reg.rt.toFixed(2)}</td>
                <td>${reg.icu_pct}%</td>
                <td>${reg.latest_cases.toLocaleString()}</td>
            `;
            tbody.appendChild(tr);

            // Map marker
            const circle = L.circleMarker([reg.lat, reg.lng], {
                radius: 12 + Math.min(18, reg.ewi_score / 4),
                color: reg.color,
                fillColor: reg.color,
                fillOpacity: 0.55,
                weight: 2
            }).addTo(state.map);

            circle.bindPopup(`
                <div style="font-family:'Plus Jakarta Sans',sans-serif; color:#111827; padding:4px;">
                    <strong style="font-size:14px;">${reg.region}</strong><br>
                    <span style="display:inline-block; margin:4px 0; padding:2px 6px; border-radius:4px; font-weight:700; font-size:11px; background:${reg.color}; color:#fff;">${reg.tier} (EWI: ${reg.ewi_score})</span><br>
                    <strong>Rt:</strong> ${reg.rt.toFixed(2)}<br>
                    <strong>ICU Load:</strong> ${reg.icu_pct}%<br>
                    <strong>Daily Cases:</strong> ${reg.latest_cases.toLocaleString()}
                </div>
            `);

            state.mapMarkers.push(circle);

            // Center map on currently selected region
            if (reg.region === state.region) {
                state.map.setView([reg.lat, reg.lng], 4);
            }
        });
    }

    // Load SitRep Briefing
    async function loadBriefing() {
        const body = document.getElementById('sitrepBody');
        body.innerHTML = '<div class="sitrep-loader"><i class="fa-solid fa-spinner fa-spin"></i> Synthesizing epidemiological situation report...</div>';

        try {
            const res = await fetch(`/api/briefing?region=${encodeURIComponent(state.region)}&pathogen=${encodeURIComponent(state.pathogen)}`);
            const rep = await res.json();

            body.innerHTML = `
                <div class="sitrep-header-block">
                    <div class="sitrep-headline" style="color:${rep.threat_color}">${rep.title}</div>
                    <div class="sitrep-meta">
                        Generated on: <strong>${rep.timestamp}</strong> &bull; Threat Alert Tier: <strong style="color:${rep.threat_color}">${rep.threat_level} (EWI ${rep.ewi_score})</strong>
                    </div>
                </div>
                <div class="sitrep-section-title">1. Executive Epidemiological Summary</div>
                <p>${rep.summary}</p>

                <div class="sitrep-section-title">2. Acute Healthcare & Resource Resilience</div>
                <p>${rep.hospital_status}</p>

                <div class="sitrep-section-title">3. Public Health Directives & Mitigation Playbook</div>
                <ul class="sitrep-recommendations">
                    ${rep.tactical_recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
            `;
        } catch (err) {
            body.innerHTML = '<p class="text-danger">Failed to generate briefing. Please try again.</p>';
        }
    }

    // Event Listeners
    document.getElementById('regionSelect').addEventListener('change', (e) => {
        state.region = e.target.value;
        loadOverview();
        loadBriefing();
    });

    document.getElementById('pathogenSelect').addEventListener('change', (e) => {
        state.pathogen = e.target.value;
        loadOverview();
        loadBriefing();
    });

    document.getElementById('horizonSelect').addEventListener('change', (e) => {
        state.horizon = parseInt(e.target.value, 10);
        loadOverview();
    });

    document.getElementById('refreshBtn').addEventListener('click', () => {
        loadOverview();
        loadBriefing();
    });

    document.getElementById('briefingBtn').addEventListener('click', () => {
        document.getElementById('sitrepSection').scrollIntoView({ behavior: 'smooth' });
        loadBriefing();
    });

    document.getElementById('copySitrepBtn').addEventListener('click', () => {
        const text = document.getElementById('sitrepBody').innerText;
        navigator.clipboard.writeText(text).then(() => {
            alert('Epidemiological Situation Briefing copied to clipboard!');
        });
    });

    // Slider Listeners with dynamic numerical update and simulation trigger
    const sliders = [
        { id: 'distRange', label: 'distVal', suffix: '%' },
        { id: 'maskRange', label: 'maskVal', suffix: '%' },
        { id: 'lockdownRange', label: 'lockdownVal', suffix: '%' },
        { id: 'vaxRateRange', label: 'vaxRateVal', transform: v => (v / 100).toFixed(2) + '%' },
        { id: 'startDayRange', label: 'startDayVal', transform: v => 'Day ' + v },
        { id: 'r0Range', label: 'r0Val', transform: v => (v / 10).toFixed(1) }
    ];

    let simDebounceTimer;
    sliders.forEach(s => {
        const el = document.getElementById(s.id);
        const lbl = document.getElementById(s.label);
        el.addEventListener('input', (e) => {
            lbl.textContent = s.transform ? s.transform(e.target.value) : e.target.value + s.suffix;
            clearTimeout(simDebounceTimer);
            simDebounceTimer = setTimeout(runSimulation, 120);
        });
    });

    document.getElementById('runSimBtn').addEventListener('click', runSimulation);

    // Initial Bootsrap
    initMap();
    initForecastChart();
    initSignalsChart();
    initSeirChart();

    loadOverview();
    runSimulation();
    loadBriefing();
});
