document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const predictionOutput = document.getElementById('predictionOutput');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form values
        const age = parseFloat(document.getElementById('age').value);
        const sex = document.getElementById('sex').value;
        const alb = parseFloat(document.getElementById('alb').value);
        const alp = parseFloat(document.getElementById('alp').value);
        const alt = parseFloat(document.getElementById('alt').value);
        const ast = parseFloat(document.getElementById('ast').value);
        const bil = parseFloat(document.getElementById('bil').value);
        const che = parseFloat(document.getElementById('che').value);
        const chol = parseFloat(document.getElementById('chol').value);
        const crea = parseFloat(document.getElementById('crea').value);
        const ggt = parseFloat(document.getElementById('ggt').value);
        const prot = parseFloat(document.getElementById('prot').value);
        
        // Validate inputs
        if (!age || !sex || !alb || !alp || !alt || !ast || !bil || !che || !chol || !crea || !ggt || !prot) {
            alert('Please fill in all fields.');
            return;
        }
        
        // Call prediction function
        const result = predictCervicalCancerRisk({
            age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot
        });
        
        displayResult(result);
    });
    
    function predictCervicalCancerRisk(patientData) {
        // This is a mock prediction function based on the features from the dataset
        // In a real implementation, this would call a trained ML model
        
        // Calculate a risk score based on the features using a weighted algorithm
        // inspired by the Random Forest model from the notebook
        let riskScore = 0;
        
        // Extract values
        const { age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot } = patientData;
        
        // Age factor - higher risk for certain age ranges
        if (age >= 30 && age <= 50) {
            riskScore += 8;
        } else if (age > 50) {
            riskScore += 12;
        }
        
        // Sex factor (female might have slightly higher risk in some studies)
        if (sex === "f") {
            riskScore += 3;
        }
        
        // Laboratory values - using weights based on typical importance in liver/kidney function
        riskScore += Math.min(10, Math.abs(alb - 40) / 2); // Normal albumin is around 40
        riskScore += Math.min(15, alp / 10); // High alkaline phosphatase
        riskScore += Math.min(15, alt / 5); // High alanine aminotransferase
        riskScore += Math.min(15, ast / 5); // High aspartate aminotransferase
        riskScore += Math.min(15, bil * 2); // High bilirubin
        riskScore += Math.min(8, Math.abs(che - 8) * 2); // Normal che is around 8
        riskScore += Math.min(8, Math.abs(chol - 5) * 2); // Normal chol is around 5
        riskScore += Math.min(10, crea / 8); // High creatinine
        riskScore += Math.min(15, ggt / 5); // High gamma-glutamyl transferase
        riskScore += Math.min(8, Math.abs(prot - 70) / 3); // Normal protein is around 70
        
        // Normalize risk score to 0-100 range
        riskScore = Math.min(100, riskScore);
        
        // Calculate risk level
        let riskLevel, riskColor, riskDescription;
        
        if (riskScore < 25) {
            riskLevel = "Low Risk (Blood Donor Category)";
            riskColor = "prediction-low";
            riskDescription = "Your values are consistent with healthy blood donor ranges. Continue maintaining a healthy lifestyle and regular checkups.";
        } else if (riskScore < 50) {
            riskLevel = "Medium Risk (Suspect Blood Donor Category)";
            riskColor = "prediction-medium";
            riskDescription = "Your values show some irregularities. Consider consulting with a healthcare professional for further evaluation and monitoring.";
        } else if (riskScore < 75) {
            riskLevel = "High Risk (Hepatitis Category)";
            riskColor = "prediction-medium";
            riskDescription = "Your values indicate possible hepatitis or liver dysfunction. Consult with a healthcare professional for proper diagnosis.";
        } else {
            riskLevel = "Very High Risk (Cirrhosis/Fibrosis Category)";
            riskColor = "prediction-high";
            riskDescription = "Your values are consistent with advanced liver disease. It is strongly recommended to consult with a healthcare professional immediately for proper diagnosis and treatment.";
        }
        
        return {
            riskLevel: riskLevel,
            riskScore: Math.round(riskScore),
            riskColor: riskColor,
            riskDescription: riskDescription
        };
    }
    
    function displayResult(result) {
        predictionOutput.innerHTML = `
            <div class="${result.riskColor}">
                <h3>Risk Level: ${result.riskLevel}</h3>
                <p>Risk Score: ${result.riskScore}/100</p>
            </div>
            <div class="prediction-info">
                <h4>Interpretation:</h4>
                <p>${result.riskDescription}</p>
            </div>
        `;
    }
});