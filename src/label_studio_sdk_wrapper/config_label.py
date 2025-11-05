COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan",
    "magenta", "lime", "teal", "indigo", "violet", "brown", "maroon", "gold",
    "silver", "navy", "coral", "salmon", "turquoise", "olive", "chocolate",
    "lavender", "khaki", "plum", "orchid", "skyblue", "crimson", "darkgreen"
]

def generate_label_tags(label_names, colors=COLORS):
    labels = []
    for i, name in enumerate(label_names):
        color = colors[i % len(colors)]  # cycle colors if >30 classes
        labels.append(f'    <Label value="{name}" background="{color}"/>')
    return "\n".join(labels)


def build_label_config(label_names):
    labels_xml = generate_label_tags(label_names)
    return f'''
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
{labels_xml}
  </RectangleLabels>
</View>
'''
