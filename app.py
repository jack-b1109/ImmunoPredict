
import gradio as gr
import pandas as pd
import joblib
import os

# --------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# --------------------------------------------------

model = joblib.load("immunopredict_model.joblib")

# --------------------------------------------------
# 2. CANCER TYPE MAPPING
# --------------------------------------------------

cancer_mapping = {
    "Bladder": 1,
    "Breast": 2,
    "Colorectal": 3,
    "Endometrial": 4,
    "Esophageal": 5,
    "Gastric": 6,
    "Head & Neck": 7,
    "Hepatobiliary": 8,
    "Melanoma": 9,
    "Mesothelioma": 10,
    "Non-small cell lung cancer (NSCLC)": 11,
    "Ovarian": 12,
    "Pancreatic": 13,
    "Renal": 14,
    "Sarcoma": 15,
    "Small cell lung cancer (SCLC)": 16
}

# --------------------------------------------------
# 3. PREDICTION FUNCTION
# --------------------------------------------------

def predict_response(tmb, therapy, albumin, nlr, age, cancer_type):

    if None in [tmb, therapy, albumin, nlr, age, cancer_type]:
        return "<p>Please complete all the input fields.</p>"

    values = {
        "TMB": tmb,
        "Systemic_therapy_history": therapy,
        "Albumin": albumin,
        "NLR": nlr,
        "Age": age
    }

    selected_type = cancer_mapping[cancer_type]

    for i in range(1, 17):
        values[f"CancerType{i}"] = 1 if selected_type == i else 0

    inputs = pd.DataFrame(
        [[values[col] for col in model.feature_names_in_]],
        columns=model.feature_names_in_
    )

    probability = float(model.predict_proba(inputs)[0][1])
    non_response_probability = 1 - probability

    threshold = 0.269
    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "Response"
        result_description = (
            "The model classifies this example as belonging to "
            "the response group."
        )
    else:
        result = "Non-response"
        result_description = (
            "The model classifies this example as belonging to "
            "the non-response group."
        )

    return f"""
    <div class="result-panel">
        <div class="result-eyebrow">MODEL OUTPUT</div>
        <h2 class="result-heading">{result}</h2>
        <p class="result-description">{result_description}</p>

        <div class="probability-grid">
            <div class="probability-card">
                <div class="probability-label">Response probability</div>
                <div class="probability-number">{probability * 100:.1f}%</div>
                <div class="bar-background">
                    <div class="bar-response"
                         style="width: {probability * 100:.1f}%"></div>
                </div>
            </div>

            <div class="probability-card">
                <div class="probability-label">Non-response probability</div>
                <div class="probability-number">
                    {non_response_probability * 100:.1f}%
                </div>
                <div class="bar-background">
                    <div class="bar-nonresponse"
                         style="width: {non_response_probability * 100:.1f}%"></div>
                </div>
            </div>
        </div>

        <div class="threshold-note">
            <strong>Decision threshold: 26.9%</strong>
            <p>
                The model uses a threshold of 26.9% to assign its
                classification. A response probability at or above
                this value is classified as Response.
            </p>
        </div>

        <div class="disclaimer">
            <strong>Important:</strong> This is an educational research
            prototype, not a clinically validated tool. Its output must
            not be used to make medical decisions or guide patient treatment.
        </div>
    </div>
    """


# --------------------------------------------------
# 4. VISUAL DESIGN
# --------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #183344;
    --navy-light: #29495A;
    --sage: #DCE9E2;
    --sage-dark: #638575;
    --cream: #F7F6F1;
    --white: #FFFFFF;
    --text: #263A43;
    --muted: #718087;
    --border: #E1E7E3;
}

.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cream) !important;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
}

#hero {
    background: var(--navy);
    color: white;
    border-radius: 22px;
    padding: 42px 44px;
    margin: 12px 0 22px 0;
}

#hero .eyebrow {
    color: #B9D2C5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

#hero h1 {
    color: white;
    font-size: 43px;
    line-height: 1.12;
    margin: 0 0 14px 0;
    font-weight: 800;
}

#hero p {
    color: #D6E1E2;
    font-size: 16px;
    line-height: 1.7;
    max-width: 720px;
}

.hero-tag {
    display: inline-block;
    background: #29495A;
    color: #DCE9E2;
    border: 1px solid #55716F;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 12px;
    margin-top: 12px;
}

.section-heading {
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    margin: 12px 0 8px 0;
}

.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 18px;
}

.info-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    height: 100%;
}

.info-card h3 {
    color: var(--navy);
    font-size: 16px;
    margin: 0 0 9px 0;
}

.info-card p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.65;
    margin: 0;
}

.input-description {
    color: #718087;
    font-size: 12px;
    line-height: 1.5;
    margin: -5px 0 12px 0;
}

.result-panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 25px;
    color: var(--text);
}

.result-eyebrow {
    color: var(--sage-dark);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
}

.result-heading {
    color: var(--navy);
    font-size: 30px;
    margin: 8px 0;
}

.result-description {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
}

.probability-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 22px 0;
}

.probability-card {
    background: #F7F8F5;
    border-radius: 12px;
    padding: 16px;
}

.probability-label {
    color: var(--muted);
    font-size: 12px;
}

.probability-number {
    color: var(--navy);
    font-size: 25px;
    font-weight: 800;
    margin: 7px 0 12px 0;
}

.bar-background {
    background: #E4EAE6;
    height: 7px;
    border-radius: 10px;
    overflow: hidden;
}

.bar-response {
    background: #638575;
    height: 7px;
    border-radius: 10px;
}

.bar-nonresponse {
    background: #8A9DA2;
    height: 7px;
    border-radius: 10px;
}

/* Threshold information: cool blue palette for clear contrast */
.threshold-note {
    background: #EAF2F8;
    border: 1px solid #C9DCEB;
    border-left: 4px solid #477A9E;
    border-radius: 8px;
    padding: 14px 16px;
    color: #183344;
    font-size: 13px;
    line-height: 1.6;
}

.threshold-note p {
    margin: 5px 0 0 0;
    color: var(--muted);
}

/* Disclaimer: neutral slate palette, visually distinct from the threshold box */
.disclaimer {
    background: #F1F3F5;
    border: 1px solid #D7DEE5;
    border-radius: 10px;
    padding: 14px;
    margin-top: 17px;
    color: #34495E;
    font-size: 12px;
    line-height: 1.6;
}

button.primary {
    background: var(--navy) !important;
    border-radius: 10px !important;
}

button.primary:hover {
    background: var(--navy-light) !important;
}

footer {
    display: none !important;
}

@media (max-width: 650px) {
    /* Use the available screen width and prevent sideways scrolling */
    .gradio-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Stack only the three overview cards; keep other layouts compact */
    #overview-cards {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        gap: 12px !important;
    }
    #overview-cards > * {
        width: 100% !important;
        min-width: 0 !important;
        flex: 0 0 auto !important;
    }

    #hero {
        padding: 26px 20px;
        margin-left: 0;
        margin-right: 0;
        border-radius: 16px;
    }
    #hero h1 {
        font-size: clamp(28px, 8vw, 33px);
    }
    #hero p {
        font-size: 14px;
    }

    .section-heading {
        font-size: 22px;
    }

    .result-panel {
        padding: 18px;
    }
    .result-heading {
        font-size: 26px;
    }
    .probability-grid {
        grid-template-columns: 1fr;
        gap: 10px;
    }

    /* Make controls easier to tap */
    .gradio-container button {
        min-height: 44px !important;
    }
    .gradio-container input,
    .gradio-container textarea {
        font-size: 16px !important;
    }

    /* Keep every tab in one horizontally scrollable navigation strip.
       Gradio can hide tabs in its overflow menu on narrow screens, so
       explicitly show the tab buttons and hide that menu control. */
    #main-tabs .tab-nav {
        display: flex !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        white-space: nowrap !important;
        gap: 6px !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }
    #main-tabs .tab-nav button,
    #main-tabs .tab-nav button[hidden],
    #main-tabs .tab-nav button[style*="display: none"] {
        display: inline-flex !important;
        flex: 0 0 auto !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: auto !important;
        min-width: max-content !important;
        max-width: none !important;
        padding: 9px 10px !important;
        font-size: 12px !important;
        line-height: 1.25 !important;
        white-space: nowrap !important;
    }
    #main-tabs .tab-nav .overflow-menu,
    #main-tabs .tab-nav [aria-label*="More"],
    #main-tabs .tab-nav [aria-label*="more"] {
        display: none !important;
    }
}
"""


# --------------------------------------------------
# 5. BUILD THE APP
# --------------------------------------------------

with gr.Blocks(
    title="ImmunoPredict | Research Prototype",
    theme=gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="green",
        neutral_hue="slate"
    ),
    css=custom_css
) as demo:

    gr.HTML("""
    <div id="hero">
        <div class="eyebrow">AI · Oncology · Research</div>
        <h1>ImmunoPredict</h1>
        <p>
            Exploring how machine learning can help identify patterns
            associated with immunotherapy response across different
            cancer types.
        </p>
        <span class="hero-tag">An educational machine learning prototype</span>
    </div>
    """)

    with gr.Tabs(elem_id="main-tabs"):

        with gr.Tab("Overview"):

            gr.HTML("""
            <div class="section-heading">Understanding the project</div>
            <div class="section-subtitle">
                ImmunoPredict is an interactive demonstration of how
                clinical and biological variables can be used in a
                machine learning classification model.
            </div>
            """)

            with gr.Row(elem_id="overview-cards"):

                gr.HTML("""
                <div class="info-card">
                    <h3>The challenge</h3>
                    <p>
                        Patients can respond differently to immune
                        checkpoint blockade. Identifying patterns
                        associated with response is an important area
                        of ongoing cancer research.
                    </p>
                </div>
                """)

                gr.HTML("""
                <div class="info-card">
                    <h3>The approach</h3>
                    <p>
                        A logistic regression model was trained on
                        published research data, using biological,
                        clinical and cancer-type variables.
                    </p>
                </div>
                """)

                gr.HTML("""
                <div class="info-card">
                    <h3>The objective</h3>
                    <p>
                        To demonstrate how machine learning can
                        explore relationships in research datasets
                        and produce interpretable classification outputs.
                    </p>
                </div>
                """)

            gr.Markdown("""
            ### Explore the model

            The predictor allows you to enter example values for five
            research variables and a cancer type. It then displays the
            model's estimated probabilities for the two outcome classes.

            **This is a research demonstration, not a medical decision tool.**
            """)

        with gr.Tab("Explore the model"):

            gr.HTML("""
            <div class="section-heading">Explore a prediction</div>
            <div class="section-subtitle">
                Enter example values below to see how the trained model
                classifies a sample. The descriptions explain each variable.
            </div>
            """)

            with gr.Row():

                with gr.Column():

                    tmb = gr.Number(
                        label="Tumour mutational burden (TMB)",
                        value=32.5
                    )
                    gr.HTML("""
                    <div class="input-description">
                        The number of mutations per megabase of tumour DNA.
                        TMB is studied as a possible indicator of how visible
                        a tumour may be to the immune system.
                    </div>
                    """)

                    therapy = gr.Radio(
                        choices=[
                            ("Yes — previous systemic therapy", 1),
                            ("No — no previous systemic therapy", 0)
                        ],
                        label="Previous systemic therapy",
                        value=1
                    )
                    gr.HTML("""
                    <div class="input-description">
                        Indicates whether the patient had received previous
                        systemic cancer treatment. This can include different
                        types of drug-based treatment.
                    </div>
                    """)

                    albumin = gr.Number(
                        label="Albumin (g/dL)",
                        value=4.3
                    )
                    gr.HTML("""
                    <div class="input-description">
                        Albumin is a protein found in blood plasma. Its level
                        can be associated with general health and inflammation.
                    </div>
                    """)

                with gr.Column():

                    nlr = gr.Number(
                        label="Neutrophil-to-lymphocyte ratio (NLR)",
                        value=1.19
                    )
                    gr.HTML("""
                    <div class="input-description">
                        The ratio of two types of white blood cells.
                        NLR is studied as a possible marker of systemic
                        inflammation and immune activity.
                    </div>
                    """)

                    age = gr.Number(
                        label="Age (years)",
                        value=67.9
                    )
                    gr.HTML("""
                    <div class="input-description">
                        Age of the example patient, expressed in years.
                    </div>
                    """)

                    cancer_type = gr.Dropdown(
                        choices=list(cancer_mapping.keys()),
                        label="Cancer type",
                        value="Bladder"
                    )
                    gr.HTML("""
                    <div class="input-description">
                        Select one of the 16 cancer types represented
                        in the research dataset.
                    </div>
                    """)

            predict_button = gr.Button(
                "Generate model output",
                variant="primary",
                size="lg"
            )

            prediction_output = gr.HTML(
                value="""
                <div class="result-panel">
                    <div class="result-eyebrow">READY TO EXPLORE</div>
                    <h2 class="result-heading">Your results will appear here</h2>
                    <p class="result-description">
                        Enter the example variables and select
                        Generate model output.
                    </p>
                </div>
                """
            )

            predict_button.click(
                fn=predict_response,
                inputs=[
                    tmb,
                    therapy,
                    albumin,
                    nlr,
                    age,
                    cancer_type
                ],
                outputs=prediction_output
            )

        with gr.Tab("Research & methodology"):

            gr.HTML("""
            <div class="section-heading">The research behind the model</div>
            <div class="section-subtitle">
                ImmunoPredict is based on the Chowell research dataset,
                which brings together clinical and biological information
                from patients treated with immune checkpoint blockade.
            </div>
            """)

            gr.Markdown("""
            ### 1. The dataset

            The project uses the Chowell training and test datasets.
            These contain patient-level variables, cancer-type information
            and recorded treatment-response outcomes.

            The model was trained using the training dataset and evaluated
            against a separate test dataset.

            ### 2. The machine learning approach

            The selected model is **logistic regression**, a supervised
            machine learning algorithm used for classification.

            It estimates the probability of an observation belonging to
            a particular outcome class based on the input variables.

            ### 3. Model performance

            The model achieved the following results on the held-out test set:

            | Evaluation metric | Result |
            |---|---:|
            | Accuracy | 69.7% |
            | ROC-AUC | 0.760 |
            | Precision | 42.7% |
            | Recall | 66.9% |
            | F1 score | 0.521 |

            These metrics describe performance on this particular dataset.
            They do not establish clinical usefulness.

            ### 4. Important limitations

            - The dataset is limited to the patients and treatments represented
              in the original research.
            - The model has not been clinically validated.
            - Its predictions may not generalise to other hospitals,
              populations or treatment settings.
            - Accuracy is lower than the 75.3% majority-class baseline
              on the test set.
            - A probability generated by the model is not a guarantee
              of an individual patient's treatment outcome.

            ### 5. Why this matters

            Machine learning may help researchers investigate complex
            relationships between tumour characteristics, patient
            variables and treatment outcomes.

            However, a model's performance must be assessed carefully
            before it can be considered for real-world clinical use.

            ---

            **Project note:** ImmunoPredict is a replication project
            inspired by the LORIS framework. It reproduces a selected
            machine-learning workflow using the publicly available
            Chowell dataset. AI tools assisted with coding and development.

            **Educational prototype only. Not for diagnosis or treatment decisions.**
            """)

    gr.Markdown("""
    ---
    <center>
    <small>ImmunoPredict · Educational research prototype · Not for clinical use</small>
    </center>
    """)


# --------------------------------------------------
# 6. LAUNCH FOR RENDER
# --------------------------------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
