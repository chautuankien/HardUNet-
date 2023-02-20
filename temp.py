for i in range(-26, -21):
    last_layer = list(model.children())[i]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = False

