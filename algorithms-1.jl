using Flux
using Flux: @epochs
using Statistics
using Random
using Parameters: @with_kw
using IterTools: ncycle 

@with_kw mutable struct Args
    lr::Float64 = 0.5
    batch_size::Int = 256
    repeat::Int = 20
end


images = Flux.Data.FashionMNIST.images()
labels = Flux.Data.FashionMNIST.labels()

function get_fashion_label(labels)
	text_label = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
	return [text_label[i+1] for i in labels]
end

#get train_data

data_x = rand(784, 6000)
data_y = []

for i in 1:6000
	data_x[:,i] = Float64.(reshape(images[i],(784,1)))
	push!(data_y, get_fashion_label(labels)[i])
end

label = sort(unique(data_y))
data_onehot_labels = Flux.onehotbatch(data_y, label)

train_x = data_x[:, [1:3:6000 ; 2:3:6000]]
train_y = data_onehot_labels[:, [1:3:6000 ; 2:3:6000]]



test_x = data_x[:, 3:3:6000]
test_y = data_onehot_labels[:, 3:3:6000]



train_data = Flux.Data.DataLoader((train_x, train_y), batchsize=Args().batch_size, shuffle=true)

#model
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10)
    )

#define loss function: cross entropy function
loss(x, y) = Flux.logitcrossentropy(model(x), y)

# params
ps = Flux.params(model)

#SDG
opt = Descent(Args().lr)

@time Flux.train!(loss, ps, ncycle(train_data, Args().repeat),opt)

accuracy(x, y, model) = Flux.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

print("train loss: ", accuracy(train_x, train_y, model), ", test loss:", accuracy(test_x, test_y, model))
