[fileName, pathName] = uigetfile('*.csv');

IMUData = dlmread([pathName fileName],',',2,0);
time = IMUData(:,1);
accData = IMUData(:,2:4);
accx=accData(:,1);
accy=accData(:,2);
accz=accData(:,3);
R = sqrt(accx.^2+accy.^2+accz.^2);

figure; subplot(2,1,1); plot(time, R,'k'); 
title('Resultant Acceleration')
xlabel('Time [s]')
ylabel('Acceleration [m/s/s]')
axis([-inf inf 0 160])
legend('Resultant acc')

subplot(2,1,2); plot(time, accData)
title('Individual Components')
xlabel('Time [s]')
ylabel('Acceleration [m/s/s]')
axis([-inf inf -100 150])
legend('Xacc','Yacc','Zacc')