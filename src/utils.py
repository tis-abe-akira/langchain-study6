def save_mermaid_to_html(mermaid_code, output_file="graph.html"):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
        <div class="mermaid">
            {mermaid_code}
        </div>
    </body>
    </html>
    """

    with open(output_file, "w") as file:
        file.write(html_content)

    print(f"Graph has been written to {output_file}. Open this file in a web browser to view the graph.")
