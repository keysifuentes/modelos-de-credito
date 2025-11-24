let chartRef = null;

// --------------------
// SIMULADOR ML
// --------------------
async function sendPredict(){
  const fields = [
    "income_monthly","payment_burden","credit_income_ratio",
    "DAYS_EMPLOYED","DAYS_BIRTH","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
    "CNT_FAM_MEMBERS","tasa_objetivo","inflacion_anual"
  ];

  const payload = {};
  fields.forEach(f=>{
    payload[f] = document.getElementById(f).value || 0;
  });

  payload["credit_type"] = document.getElementById("credit_type").value;

  // 1) predicción normal
  const res = await fetch("/predict",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });

  const out = await res.json();
  document.getElementById("result").innerHTML =
    `<h3>Resultado</h3>
     <p><b>PD (probabilidad de default):</b> ${out.pd_default}</p>
     <p><b>Decisión:</b> ${out.decision}</p>
     <p><b>Umbral por producto:</b> ${out.threshold_used}</p>`;

  // 2) gráfica dinámica de sensibilidad
  await updateSensitivityChart(payload);
}

// --------------------
// GRÁFICA DINÁMICA PD vs TASA
// --------------------
async function updateSensitivityChart(payload){
  // pedimos al backend el barrido de tasas
  const res = await fetch("/sensitivity",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({
      ...payload,
      rate_min: 5,
      rate_max: 15,
      steps: 11
    })
  });

  const out = await res.json();
  const rates = out.rates;
  const pds = out.pds;

  const ctx = document.getElementById("pdRateChart");

  // si ya existe, destruir para redibujar
  if(chartRef) chartRef.destroy();

  chartRef = new Chart(ctx, {
    type: "line",
    data: {
      labels: rates,
      datasets: [{
        label: "PD estimada",
        data: pds,
        tension: 0.25
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display:true, text:"Tasa Banxico (%)"} },
        y: { title: { display:true, text:"Probabilidad de Default (PD)"} , min:0, max:1}
      }
    }
  });
}

