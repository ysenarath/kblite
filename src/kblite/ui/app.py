import os

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from kblite.analyze import analyze_text, get_scores

st.title("Text Analysis")

# Sample text options
samples = {
    "Philosophy": "The philosopher contemplated the nature of consciousness and free will. She questioned whether artificial intelligence could develop genuine emotions or if consciousness requires biological foundations. The debate between determinism and free will remained unresolved.",
    "Science": "Quantum entanglement demonstrates the peculiar nature of reality at the microscopic level. Scientists observed how particles maintain instantaneous correlations across vast distances, challenging our understanding of causality and local realism. The wave-particle duality of light further complicates our model of the universe.",
    "Economics": "The global economy experienced inflation as supply chain disruptions affected market dynamics. Central banks adjusted interest rates while investors analyzed market trends. The cryptocurrency ecosystem introduced new paradigms in digital finance and decentralized economics.",
    "Psychology": "The therapist explored how childhood trauma influences adult behavior patterns and emotional regulation. Cognitive behavioral therapy techniques helped patients develop resilience and overcome anxiety. The relationship between nature and nurture shaped personality development.",
    "Literature": "The novel explored themes of existentialism through its protagonist's journey of self-discovery. Metaphors and symbolism wove together narratives of love, loss, and redemption. The author's stream of consciousness style reflected the character's internal struggles.",
}

# Sample selector
sample_option = st.selectbox(
    "Choose a sample text or enter your own:", ["Custom"] + list(samples.keys())
)

# Text input area
text = st.text_area(
    "Enter text to analyze:",
    value=samples[sample_option] if sample_option != "Custom" else "",
    height=200,
)

if text:
    # Process results
    results = list(analyze_text(text, max_cost=2))
    results = get_scores(results)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Interactive Network",
            "Term Details",
            "Text Analysis",
            "Processor",
        ]
    )

    # Group results by term
    term_groups = {}
    for term, start, end, triple, score in results:
        if term not in term_groups:
            term_groups[term] = set()
        term_groups[term].add(triple)  # triple is already a set

    with tab1:
        st.subheader("Knowledge Network")

        # Create network
        net = Network(
            height="600px", width="100%", bgcolor="#ffffff", font_color="black"
        )

        # Configure physics
        net.force_atlas_2based(
            gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08
        )

        # Add nodes and edges
        for term, contexts_set in term_groups.items():
            print(contexts_set)
            # Add term node
            net.add_node(
                str(term), str(term), title=str(term), color="#add8e6"
            )  # light blue

            # Flatten contexts_set since it's a set of sets
            for sub, rel, ent in contexts_set:
                # Add context node
                net.add_node(str(ent), str(ent), title=str(ent), color="#90ee90")
                net.add_edge(
                    str(term), str(ent), title=f"{term}-{rel}->{ent}", color="#000000"
                )

        # Generate and save the html file
        html_path = "network.html"
        net.save_graph(html_path)

        # Read the html file and embed it
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)

        # Clean up
        if os.path.exists(html_path):
            os.remove(html_path)

        st.caption("💡 Tip: Drag nodes to rearrange, scroll to zoom, hover for details")

    with tab2:
        st.subheader("Found Terms")
        # Display each term and its contexts
        for term, contexts_set in term_groups.items():
            with st.expander(f"📝 {term}"):
                st.write("Related concepts:")
                # Flatten contexts_set
                all_contexts = {str(contexts) for contexts in contexts_set}
                for context in all_contexts:
                    st.write(f"- {context}")

    with tab3:
        st.subheader("Text Analysis")
        # Highlight terms in text
        highlighted_text = text
        # Sort by start position in reverse to avoid index shifting
        for term, start, end, _, _ in sorted(results, key=lambda x: x[1], reverse=True):
            highlighted_text = (
                highlighted_text[:start]
                + f"**{highlighted_text[start:end]}**"
                + highlighted_text[end:]
            )
        st.markdown(highlighted_text)

    with tab4:
        st.subheader("Processor")
        # Display the processor code
        st.write(results)
