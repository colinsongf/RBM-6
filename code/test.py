from Experience import Experience

model = Experience(1000,'log3')
model.fine_tune(epochs=12, lr=0.01, dropout=True, lcost=True)
