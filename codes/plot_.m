train_loss_matrix = reshape(train_loss,782,30);
train_loss_epoch = mean(train_loss_matrix);
figure(1);plot(train_loss_epoch)
figure(2);
train_accuracy_matrix = reshape(train_accuracy, 782,30);
train_accuracy_epoch = mean(train_accuracy_matrix);
plot(train_accuracy_epoch);
hold on;
%test_accuracy(23)=0.61;
plot(test_accuracy);