window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("predict-btn");
  const resultDiv = document.getElementById("result");

  btn.addEventListener("click", async () => {
    const rawId = document.getElementById("product_id").value;
    // Cast to integer
    const prodId = parseInt(rawId, 10);
    if (isNaN(prodId)) {
      resultDiv.textContent = "Please enter a valid integer Product ID.";
      return;
    }

    resultDiv.textContent = "Loading...";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: prodId }),
      });
      const data = await res.json();
      if (res.ok) {
        resultDiv.innerHTML = `
                    <p><strong>Product ID:</strong> ${data.product_id}</p>
                    <p><strong>Prob to buy again:</strong> ${data.to_buy_probability.toFixed(
                      3
                    )}</p>
                    <p><strong>Prediction:</strong> ${
                      data.to_buy_prediction ? "Yes" : "No"
                    }</p>
                `;
      } else {
        resultDiv.textContent = data.error;
      }
    } catch (e) {
      resultDiv.textContent = "Error: " + e;
    }
  });
});
