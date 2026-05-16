/* VIC Fuel Forecast Dashboard — main.js
   CL04_G04 · Swinburne COS40007 AI Engineering */

/* CONFIG  */
const GITHUB_ORG    = 'COS40007-2026-Classrooms';
const GITHUB_REPO   = 'CL04_G04';
const GITHUB_BRANCH = 'main';
const RAW = `https://raw.githubusercontent.com/${GITHUB_ORG}/${GITHUB_REPO}/${GITHUB_BRANCH}`;
const MAPBOX_TOKEN  ='pk.eyJ1IjoiZ2F2LXRlc3QtYXJjIiwiYSI6ImNtb3JkMG10dzAyMDUycW9wa2h0bzlnbGcifQ.VsQylzIqJDNoSX6DbGAxOg';

const C = {
  c1:'#0B7F8C', c2:'#2CBFBF', c3:'#F2A413', c4:'#D96704', c5:'#BF3604',
  bdr:'#E2E8ED', txt:'#0A0A0A', txt3:'#718096'
};
const ZONE_C = {
  inner_metro:  C.c1,
  middle_metro: C.c3,
  regional_vic: C.c5,
  rural_remote: '#888'
};

let suburbData  = null;
let forecastData = null;
let mapInstance = null;
let curMapFuel  = 'ulp91';

/*  POSTCODE CENTROIDS  
I used AI help in getting the coordinates */

const PC = {
  '3000':[-37.8136,144.9631],'3002':[-37.8153,144.9842],'3003':[-37.7987,144.9444],
  '3006':[-37.8230,144.9575],'3008':[-37.8178,144.9394],'3011':[-37.8007,144.8997],
  '3012':[-37.8275,144.8603],'3013':[-37.8156,144.8844],'3015':[-37.8530,144.8832],
  '3016':[-37.8623,144.8966],'3018':[-37.8694,144.8272],'3019':[-37.8432,144.8439],
  '3020':[-37.7931,144.8289],'3021':[-37.7718,144.8264],'3022':[-37.7598,144.7994],
  '3023':[-37.7765,144.7694],'3024':[-37.9272,144.7019],'3025':[-37.8643,144.8055],
  '3027':[-37.8656,144.7485],'3028':[-37.8810,144.7002],'3029':[-37.8946,144.7003],
  '3030':[-37.9067,144.7461],'3031':[-37.7889,144.9325],'3032':[-37.7634,144.9158],
  '3033':[-37.7537,144.8820],'3034':[-37.7280,144.8262],'3036':[-37.7427,144.8021],
  '3037':[-37.7228,144.7774],'3038':[-37.7178,144.8168],'3039':[-37.7700,144.9194],
  '3040':[-37.7375,144.9216],'3041':[-37.7239,144.9305],'3042':[-37.7265,144.8757],
  '3043':[-37.7008,144.8861],'3044':[-37.7058,144.9418],'3046':[-37.6987,144.9486],
  '3047':[-37.6777,144.9499],'3048':[-37.6733,144.9325],'3049':[-37.6671,144.9141],
  '3058':[-37.7430,144.9671],'3063':[-37.5951,144.9445],'3064':[-37.5962,144.9235],
  '3072':[-37.7512,145.0105],'3073':[-37.7330,145.0119],'3074':[-37.7200,145.0174],
  '3075':[-37.7057,145.0210],'3076':[-37.6762,145.0131],'3082':[-37.6869,145.0587],
  '3083':[-37.7085,145.0720],'3084':[-37.7192,145.0906],'3106':[-37.7835,145.1024],
  '3130':[-37.8162,145.1554],'3134':[-37.7950,145.2292],'3136':[-37.8066,145.2772],
  '3149':[-37.8713,145.1322],'3150':[-37.8787,145.1617],'3152':[-37.8748,145.2220],
  '3155':[-37.8832,145.2839],'3156':[-37.8770,145.2647],'3170':[-37.9191,145.1614],
  '3173':[-37.9680,145.1782],'3174':[-37.9469,145.1872],'3175':[-37.9885,145.2091],
  '3177':[-37.9564,145.2398],'3178':[-37.9262,145.2398],'3195':[-38.0111,145.1075],
  '3199':[-38.1450,145.1249],'3201':[-38.1089,145.1555],'3216':[-38.1681,144.3814],
  '3217':[-38.2001,144.3498],'3280':[-38.3793,142.4851],'3335':[-37.6636,144.5527],
  '3336':[-37.6781,144.5268],'3337':[-37.6913,144.5553],'3338':[-37.6756,144.5956],
  '3340':[-37.7043,144.4262],'3350':[-37.5622,143.8502],'3429':[-37.5762,144.7286],
  '3500':[-34.1851,142.1614],'3550':[-36.7580,144.2793],'3551':[-36.7956,144.3076],
  '3630':[-36.3805,145.4049],'3677':[-36.3635,146.3048],'3690':[-36.1213,146.8922],
  '3750':[-37.6558,145.1071],'3751':[-37.6239,145.1527],'3752':[-37.6112,145.1374],
  '3754':[-37.5942,145.1748],'3755':[-37.5785,145.1349],'3756':[-37.4192,144.9573],
  '3800':[-37.9132,145.1339],'3802':[-37.9480,145.2398],'3803':[-37.9890,145.2719],
  '3804':[-38.0237,145.2962],'3805':[-38.0514,145.2748],'3806':[-38.0370,145.3492],
  '3810':[-38.0716,145.4918],'3820':[-38.1644,145.9334],'3825':[-38.1738,146.2683],
  '3840':[-38.1993,146.5276],'3842':[-38.2451,146.4196],'3844':[-38.2053,146.5420],
  '3850':[-38.1048,147.0659],'3875':[-37.8399,147.6151],'3910':[-38.1697,145.1830],
  '3975':[-38.1142,145.3286],'3976':[-38.1529,145.2777],'3977':[-38.1254,145.3082],
  '3978':[-38.1041,145.2612],'3980':[-38.1991,145.5212],
};

/*  TOOLTIP  */
const tip = document.getElementById('tip');
function showTip(html, e) { tip.innerHTML = html; tip.style.display = 'block'; moveTip(e); }
function moveTip(e) { tip.style.left = (e.clientX + window.scrollX + 12) + 'px'; tip.style.top = (e.clientY + window.scrollY - 10) + 'px'; }
function hideTip() { tip.style.display = 'none'; }

/*  LOAD DATA  */
async function loadData() {
  try {
    const [s, f] = await Promise.all([
      fetch(`${RAW}/reports/suburb_price_forecast.json`),
      fetch(`${RAW}/reports/price_forecast.json`)
    ]);
    suburbData   = await s.json();
    forecastData = await f.json();
  } catch {
    suburbData   = demoSuburbs();
    forecastData = demoForecast();
  }
  const gen = new Date(suburbData.generated_at || Date.now());
  document.getElementById('lup').textContent = 'Updated ' + gen.toLocaleDateString('en-AU', {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
  });
  rSum();
  initSearch();
  initMap();
  rGroupedBar();
  rBollinger();
  rScatter();
  rTop3();
}

/*  UTILS  */
function gs(id) { document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
function setFB(b, fuel) {
  document.querySelectorAll('.ftog button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  curMapFuel = fuel;
  buildMapLayer(fuel);
}

/*  SUMMARY CARDS  */
function rSum() {
  const sf = forecastData.state_forecasts || suburbData.state_forecasts || {};
  document.getElementById('sg').innerHTML = ['ulp91', 'ulp95', 'diesel'].map(k => {
    const d   = sf[k] || {};
    const chg = d.change_cpl || 0;
    const dir = chg < -.5 ? 'dn' : chg > .5 ? 'up' : 'fl';
    const arr = dir === 'dn' ? '↓' : dir === 'up' ? '↑' : '→';
    const cls = { ulp91: 'u91', ulp95: 'u95', diesel: 'dsl' }[k];
    const lbl = { ulp91: 'Regular Unleaded 91', ulp95: 'Premium Unleaded 95', diesel: 'Diesel' }[k];
    return `<div class="fc ${cls}">
      <div class="cl">${lbl}</div>
      <div style="display:flex;align-items:baseline;gap:7px;margin-bottom:8px">
        <div class="cp">${(d.current_cpl || 0).toFixed(1)}</div>
        <div class="cu">cpl now</div>
      </div>
      <div class="cr">
        <div>
          <div class="cfl">Next week forecast</div>
          <div class="cfv">${(d.forecast_cpl || 0).toFixed(1)} cpl</div>
        </div>
        <div class="cc ${dir}">${arr} ${chg > 0 ? '+' : ''}${chg.toFixed(1)}</div>
      </div>
    </div>`;
  }).join('');
}

/*  SEARCH  */
function initSearch() {
  const inp  = document.getElementById('si');
  const res  = document.getElementById('sr');
  const subs = suburbData.suburbs || [];

  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    if (q.length < 2) { res.style.display = 'none'; return; }
    const m = subs.filter(s =>
      s.suburb_name.toLowerCase().includes(q) || s.postcode.includes(q)
    ).slice(0, 8);
    if (!m.length) { res.style.display = 'none'; return; }
    res.innerHTML = m.map(s => `
      <div class="sri" onclick="selectSub('${s.postcode}','${s.suburb_name}')">
        <div>
          <div class="srn">${s.suburb_name}</div>
          <div class="srp">${s.postcode} · ${(s.zone || '').replace('_', ' ')}</div>
        </div>
        <div class="srv">${s.fuels?.ulp91?.forecast_cpl?.toFixed(1) || '—'} cpl</div>
      </div>`).join('');
    res.style.display = 'block';
  });
  document.addEventListener('click', e => { if (!e.target.closest('#ss')) res.style.display = 'none'; });
}

function selectSub(pc, name) {
  document.getElementById('si').value = name;
  document.getElementById('sr').style.display = 'none';
  flyTo(pc);
}

function flyTo(pc) {
  const c = PC[pc];
  if (!c || !mapInstance) return;
  mapInstance.flyTo({ center: [c[1], c[0]], zoom: 12, duration: 1200 });
  document.getElementById('map').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/*  MAP  */
function initMap() {
  if (MAPBOX_TOKEN === '__MAP_API__' || typeof mapboxgl === 'undefined') {
    document.getElementById('map').innerHTML = `
      <div style="height:520px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:#718096;font-size:13px">
        <div style="font-size:42px;opacity:.25">🗺</div>
        <div style="text-align:center">
          <strong style="color:#0B7F8C;display:block;margin-bottom:6px">Mapbox map</strong>
          Requires MAP_API — injected at deploy time
        </div>
      </div>`;
    return;
  }
  mapboxgl.accessToken = MAPBOX_TOKEN;
  mapInstance = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [145.3, -37.1],
    zoom: 7.2,
    maxBounds: [[140, -40], [151, -33.5]]
  });
  mapInstance.addControl(new mapboxgl.NavigationControl(), 'top-right');
  mapInstance.on('load', () => buildMapLayer('ulp91'));
}

function buildMapLayer(fuel) {
  if (!mapInstance) return;
  const subs = (suburbData.suburbs || []).map(s => {
    const c  = PC[s.postcode];
    if (!c) return null;
    const fd = s.fuels?.[fuel] || {};
    return {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [c[1], c[0]] },
      properties: {
        postcode:    s.postcode,
        suburb_name: s.suburb_name,
        zone:        s.zone,
        forecast:    fd.forecast_cpl || 0,
        u91:         (s.fuels?.ulp91  || {}).forecast_cpl || 0,
        u95:         (s.fuels?.ulp95  || {}).forecast_cpl || 0,
        dsl:         (s.fuels?.diesel || {}).forecast_cpl || 0,
      }
    };
  }).filter(Boolean);

  const prices = subs.map(f => f.properties.forecast).filter(v => v > 0);
  const mn  = Math.min(...prices);
  const mx  = Math.max(...prices);
  const mid = (mn + mx) / 2;

  document.getElementById('mln').textContent = mn.toFixed(0);
  document.getElementById('mlx').textContent = mx.toFixed(0);

  const expr = ['interpolate', ['linear'], ['get', 'forecast'],
    mn, '#2CBFBF', mid, '#F2A413', mx, '#BF3604'];

  if (mapInstance.getSource('s')) {
    mapInstance.getSource('s').setData({ type: 'FeatureCollection', features: subs });
    mapInstance.setPaintProperty('sc', 'circle-color', expr);
  } else {
    mapInstance.addSource('s', { type: 'geojson', data: { type: 'FeatureCollection', features: subs } });
    mapInstance.addLayer({
      id: 'sc', type: 'circle', source: 's',
      paint: {
        'circle-radius':        ['interpolate', ['linear'], ['zoom'], 6, 5, 11, 13],
        'circle-color':         expr,
        'circle-opacity':       .85,
        'circle-stroke-width':  1.5,
        'circle-stroke-color':  'white',
        'circle-stroke-opacity': .6
      }
    });
    const popup = new mapboxgl.Popup({ closeButton: false, maxWidth: '210px' });
    mapInstance.on('mouseenter', 'sc', e => {
      mapInstance.getCanvas().style.cursor = 'pointer';
      const p = e.features[0].properties;
      popup.setLngLat(e.lngLat).setHTML(`
        <div class="pn">${p.suburb_name}</div>
        <div class="pp">${p.postcode}</div>
        <div class="pr"><span class="prl">ULP91</span><span class="prv" style="color:#0B7F8C">${p.u91.toFixed(1)} cpl</span></div>
        <div class="pr"><span class="prl">ULP95</span><span class="prv" style="color:#9A6800">${p.u95.toFixed(1)} cpl</span></div>
        <div class="pr"><span class="prl">Diesel</span><span class="prv" style="color:#BF3604">${p.dsl.toFixed(1)} cpl</span></div>
        <span class="pz">${(p.zone || '').replace(/_/g, ' ')}</span>
      `).addTo(mapInstance);
    });
    mapInstance.on('mouseleave', 'sc', () => { mapInstance.getCanvas().style.cursor = ''; popup.remove(); });
  }
}

/* =============================================================================
   D3 CHARTS
============================================================================= */

/*  1. GROUPED BAR: Current vs Forecast  */
function rGroupedBar() {
  const sf     = forecastData.state_forecasts || suburbData.state_forecasts || {};
  const fuels  = ['ulp91', 'ulp95', 'diesel'];
  const labels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };
  const data   = fuels.map(k => ({
    fuel:     labels[k],
    current:  (sf[k] || {}).current_cpl  || 0,
    forecast: (sf[k] || {}).forecast_cpl || 0,
    change:   (sf[k] || {}).change_cpl   || 0
  }));

  const W  = document.getElementById('chart-grouped').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 40, left: 50 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const x0 = d3.scaleBand().domain(data.map(d => d.fuel)).range([0, iW]).paddingInner(.25).paddingOuter(.1);
  const x1 = d3.scaleBand().domain(['current', 'forecast']).range([0, x0.bandwidth()]).padding(.05);
  const allVals = data.flatMap(d => [d.current, d.forecast]);
  const y  = d3.scaleLinear().domain([d3.min(allVals) * .98, d3.max(allVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-grouped').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  const fuelColors = { ULP91: C.c1, ULP95: C.c3, Diesel: C.c5 };

  data.forEach(d => {
    const col = fuelColors[d.fuel];
    const grp = g.append('g').attr('transform', `translate(${x0(d.fuel)},0)`);

    grp.append('rect')
      .attr('x', x1('current')).attr('y', y(d.current))
      .attr('width', x1.bandwidth()).attr('height', iH - y(d.current))
      .attr('fill', col).attr('opacity', .3).attr('rx', 3)
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Current: ${d.current.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);

    grp.append('rect')
      .attr('x', x1('forecast')).attr('y', y(d.forecast))
      .attr('width', x1.bandwidth()).attr('height', iH - y(d.forecast))
      .attr('fill', col).attr('opacity', .9).attr('rx', 3)
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Forecast: ${d.forecast.toFixed(1)} cpl<br>Change: ${d.change > 0 ? '+' : ''}${d.change.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);

    const chgColor = d.change < 0 ? '#1A7A4A' : '#B85C00';
    grp.append('text')
      .attr('x', x0.bandwidth() / 2).attr('y', y(Math.max(d.current, d.forecast)) - 6)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600)
      .attr('fill', chgColor)
      .text(`${d.change > 0 ? '+' : ''}${d.change.toFixed(1)}`);
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x0).tickSize(0)).select('.domain').remove();
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', -iH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('cents per litre');
}

/*  2. BOLLINGER BANDS: Suburb price distribution  */
function rBollinger() {
  const subs      = suburbData.suburbs || [];
  const fuels     = ['ulp91', 'ulp95', 'diesel'];
  const fuelLabels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };
  const fuelCols   = { ulp91: C.c1,    ulp95: C.c3,    diesel: C.c5    };

  const bands = fuels.map(k => {
    const vals = subs.map(s => s.fuels?.[k]?.forecast_cpl).filter(v => v && v > 0).sort(d3.ascending);
    return {
      fuel:   fuelLabels[k],
      key:    k,
      min:    d3.min(vals),
      q1:     d3.quantile(vals, .25),
      median: d3.median(vals),
      q3:     d3.quantile(vals, .75),
      max:    d3.max(vals)
    };
  });

  const W  = document.getElementById('chart-bollinger').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 40, left: 50 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const x       = d3.scaleBand().domain(fuels.map(k => fuelLabels[k])).range([0, iW]).padding(.4);
  const allVals = bands.flatMap(d => [d.min, d.max]);
  const y       = d3.scaleLinear().domain([d3.min(allVals) * .97, d3.max(allVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-bollinger').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  bands.forEach(d => {
    const cx  = x(d.fuel) + x.bandwidth() / 2;
    const col = fuelCols[d.key];

    // Whisker
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', y(d.min)).attr('y2', y(d.max))
      .attr('stroke', col).attr('stroke-width', 1.5).attr('stroke-dasharray', '3,3').attr('opacity', .5);

    // IQR box
    g.append('rect')
      .attr('x', x(d.fuel)).attr('y', y(d.q3))
      .attr('width', x.bandwidth()).attr('height', y(d.q1) - y(d.q3))
      .attr('fill', col).attr('opacity', .18).attr('rx', 4);

    // Median line
    g.append('line')
      .attr('x1', x(d.fuel)).attr('x2', x(d.fuel) + x.bandwidth())
      .attr('y1', y(d.median)).attr('y2', y(d.median))
      .attr('stroke', col).attr('stroke-width', 2.5);

    // Min/Max caps
    [d.min, d.max].forEach(v => {
      g.append('line')
        .attr('x1', cx - 8).attr('x2', cx + 8)
        .attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', col).attr('stroke-width', 1.5).attr('opacity', .6);
    });

    // Median label
    g.append('text')
      .attr('x', cx).attr('y', y(d.median) - 7)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('font-weight', 600).attr('fill', col)
      .text(d.median?.toFixed(1));

    // Hover target
    g.append('rect')
      .attr('x', x(d.fuel) - 8).attr('y', y(d.max) - 4)
      .attr('width', x.bandwidth() + 16).attr('height', y(d.min) - y(d.max) + 8)
      .attr('fill', 'transparent')
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Max: ${d.max?.toFixed(1)} cpl<br>Q3: ${d.q3?.toFixed(1)} cpl<br>Median: ${d.median?.toFixed(1)} cpl<br>Q1: ${d.q1?.toFixed(1)} cpl<br>Min: ${d.min?.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x).tickSize(0)).select('.domain').remove();
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', -iH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('cents per litre');

  document.getElementById('bb-legend').innerHTML = `
    <div class="cli"><div style="width:28px;height:3px;background:#888"></div>Range</div>
    <div class="cli"><div style="width:16px;height:10px;background:var(--c2);opacity:.3;border-radius:2px"></div>IQR</div>
    <div class="cli"><div style="width:16px;height:2.5px;background:var(--c2)"></div>Median</div>`;
}

/*  3. SCATTER: ULP91 forecast vs petrol vehicle count  */
function rScatter() {
  const subs = (suburbData.suburbs || [])
    .filter(s => s.fuels?.ulp91?.forecast_cpl && s.petrol_vehicles > 0)
    .slice(0, 120);

  const W  = document.getElementById('chart-scatter').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 44, left: 54 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const x     = d3.scaleLinear().domain([0, d3.max(subs, d => d.petrol_vehicles) * 1.05]).range([0, iW]);
  const yVals = subs.map(d => d.fuels.ulp91.forecast_cpl);
  const y     = d3.scaleLinear().domain([d3.min(yVals) * .98, d3.max(yVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-scatter').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  // Regression line
  const xm  = d3.mean(subs, d => d.petrol_vehicles);
  const ym  = d3.mean(subs, d => d.fuels.ulp91.forecast_cpl);
  const num = d3.sum(subs, d => (d.petrol_vehicles - xm) * (d.fuels.ulp91.forecast_cpl - ym));
  const den = d3.sum(subs, d => (d.petrol_vehicles - xm) ** 2);
  const slope = num / den;
  const intercept = ym - slope * xm;
  const x1r = d3.min(subs, d => d.petrol_vehicles);
  const x2r = d3.max(subs, d => d.petrol_vehicles);

  g.append('line')
    .attr('x1', x(x1r)).attr('y1', y(slope * x1r + intercept))
    .attr('x2', x(x2r)).attr('y2', y(slope * x2r + intercept))
    .attr('stroke', C.txt3).attr('stroke-width', 1).attr('stroke-dasharray', '4,4').attr('opacity', .6);

  // Dots
  subs.forEach(d => {
    const col = ZONE_C[d.zone] || '#888';
    g.append('circle')
      .attr('cx', x(d.petrol_vehicles)).attr('cy', y(d.fuels.ulp91.forecast_cpl))
      .attr('r', 4).attr('fill', col).attr('opacity', .72)
      .attr('stroke', '#fff').attr('stroke-width', .8)
      .style('cursor', 'pointer')
      .on('mouseover', e => showTip(`<strong>${d.suburb_name}</strong>${d.postcode}<br>ULP91: ${d.fuels.ulp91.forecast_cpl.toFixed(1)} cpl<br>Petrol vehicles: ${(d.petrol_vehicles || 0).toLocaleString()}`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip)
      .on('click', () => flyTo(d.postcode));
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v));
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', iW / 2).attr('y', iH + 36)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('Petrol vehicles registered');
  g.append('text').attr('x', -iH / 2).attr('y', -40).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('ULP91 forecast (cpl)');
}

/*  TOP 3  */
function rTop3() {
  const sf   = forecastData.state_forecasts || suburbData.state_forecasts || {};
  const subs = suburbData.suburbs || [];
  document.getElementById('t3g').innerHTML = ['ulp91', 'ulp95', 'diesel'].map(k => {
    const sorted = [...subs].filter(s => s.fuels?.[k]?.forecast_cpl)
      .sort((a, b) => a.fuels[k].forecast_cpl - b.fuels[k].forecast_cpl).slice(0, 3);
    const avg = (sf[k] || {}).forecast_cpl || 0;
    const cls = { ulp91: 'u91', ulp95: 'u95', diesel: 'dsl' }[k];
    const lbl = { ulp91: 'Cheapest ULP91', ulp95: 'Cheapest ULP95', diesel: 'Cheapest Diesel' }[k];
    return `<div class="t3c">
      <div class="t3h"><span class="t3b ${cls}">${k.toUpperCase()}</span><span class="t3tl">${lbl} next week</span></div>
      ${sorted.map((s, i) => {
        const fc = s.fuels[k].forecast_cpl;
        const sv = avg - fc;
        return `<div class="t3i" onclick="flyTo('${s.postcode}')">
          <div class="rb r${i + 1}">${i + 1}</div>
          <div class="t3s">
            <div class="t3sn">${s.suburb_name}</div>
            <div class="t3sz">${s.postcode} · ${(s.zone || '').replace('_', ' ')}</div>
          </div>
          <div style="text-align:right">
            <div class="t3pv">${fc.toFixed(1)}</div>
            <div class="t3pu">cpl</div>
            ${sv > 0 ? `<div class="t3sv">↓ ${sv.toFixed(1)} below avg</div>` : ''}
          </div>
        </div>`;
      }).join('')}
    </div>`;
  }).join('');
}

/*  DEMO DATA  */
function demoForecast() {
  return { generated_at: new Date().toISOString(), state_forecasts: {
    ulp91:  { current_cpl: 180.9, forecast_cpl: 171.9, change_cpl: -9.0, forecast_date: '2026-05-08', week_end: '2026-05-14' },
    ulp95:  { current_cpl: 195.9, forecast_cpl: 188.9, change_cpl: -7.0, forecast_date: '2026-05-08', week_end: '2026-05-14' },
    diesel: { current_cpl: 253.5, forecast_cpl: 258.4, change_cpl:  4.9, forecast_date: '2026-05-08', week_end: '2026-05-14' }
  }};
}

function demoSuburbs() {
  const rows = [
    ['3023','Deer Park','inner_metro',173.9,163.9,188.9,178.9,246.5,251.4,39034,10599],
    ['3338','Melton','middle_metro',174.9,164.9,189.9,179.9,247.5,252.4,25662,7487],
    ['3429','Sunbury','middle_metro',174.9,164.9,189.9,179.9,247.5,252.4,26796,9400],
    ['3029','Hoppers Crossing','inner_metro',179.9,169.9,194.9,184.9,250.5,255.4,91799,25678],
    ['3350','Ballarat','middle_metro',176.9,166.9,191.9,181.9,248.5,253.4,42968,15497],
    ['3216','Geelong','middle_metro',177.2,167.2,192.2,182.2,249.5,254.4,36102,11380],
    ['3550','Bendigo','regional_vic',181.3,171.3,196.3,186.3,253.5,258.4,27567,11441],
    ['3500','Mildura','middle_metro',195.9,185.9,210.9,200.9,263.5,268.4,22327,12065],
    ['3630','Shepparton','regional_vic',179.9,169.9,194.9,184.9,251.5,256.4,21916,11140],
    ['3690','Wodonga','regional_vic',180.9,170.9,195.9,185.9,252.5,257.4,24976,13841],
    ['3030','Point Cook','inner_metro',180.8,171.8,195.8,185.8,253.4,258.3,71566,19663],
    ['3977','Cranbourne South','regional_vic',179.9,169.9,194.9,184.9,252.5,257.4,71613,22795],
    ['3064','Craigieburn','inner_metro',179.9,169.9,194.9,184.9,252.5,257.4,77467,20588],
    ['3175','Dandenong','inner_metro',181.7,171.9,196.7,186.9,254.3,258.4,40105,21589],
    ['3844','Traralgon','regional_vic',177.4,168.4,192.4,183.4,251.0,255.9,22253,11531],
    ['3810','Pakenham','regional_vic',177.9,168.9,192.9,183.9,251.5,256.4,35268,13865],
    ['3805','Narre Warren South','regional_vic',177.4,168.4,192.4,183.4,251.0,255.9,35803,10141],
    ['3806','Berwick','regional_vic',176.8,167.8,191.8,182.8,250.4,255.3,33992,9757],
    ['3136','Croydon','inner_metro',178.3,169.3,193.3,184.3,251.9,256.8,28353,7443],
    ['3073','Reservoir','inner_metro',178.9,169.9,193.9,184.9,252.5,257.4,29403,5830],
  ];
  const sf = {
    ulp91:  { current_cpl: 180.9, forecast_cpl: 171.9, change_cpl: -9.0 },
    ulp95:  { current_cpl: 195.9, forecast_cpl: 188.9, change_cpl: -7.0 },
    diesel: { current_cpl: 253.5, forecast_cpl: 258.4, change_cpl:  4.9 }
  };
  return { generated_at: new Date().toISOString(), state_forecasts: sf,
    suburbs: rows.map(d => ({
      postcode: d[0], suburb_name: d[1], zone: d[2],
      petrol_vehicles: d[10], diesel_vehicles: d[11],
      fuels: {
        ulp91:  { current_cpl: d[3], forecast_cpl: d[4], change_cpl: d[4] - d[3] },
        ulp95:  { current_cpl: d[5], forecast_cpl: d[6], change_cpl: d[6] - d[5] },
        diesel: { current_cpl: d[7], forecast_cpl: d[8], change_cpl: d[8] - d[7] }
      }
    }))
  };
}

/*  BOOT  */
window.addEventListener('load', loadData);
