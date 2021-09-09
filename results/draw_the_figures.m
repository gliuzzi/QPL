clear all
close all
clc

load('WKS_RESULTS.mat')
colormap parula


%%%%%%%%%%%%%% dim. 20+30+40+50
hf = figure('Position',[0,0,1000,1000]);
stairs(sort(BEST2_10),[1:64]/64,'s-','Linewidth',1.2,'Markersize',10)
hold on
stairs(sort(QIP),[1:64]/64,'+-','Linewidth',1.2,'Markersize',10)
stairs(sort(BARON),[1:64]/64,'*-','Linewidth',1.2,'Markersize',10)
stairs(sort(GUROBI),[1:64]/64,'d-','Linewidth',1.2,'Markersize',10)
stairs(sort(CPLEX),[1:64]/64,'x-','Linewidth',1.2,'Markersize',10)
xlim([0,30])
xlabel('Time (in seconds)')
ylabel('Fraction of problems solved')
title('Problems with n=20,30,40,50')
legend('B&T(Mix)','QuadprogIP','BARON','GUROBI','CPLEX','Location','southeast')
saveas(hf,['CompareSolvers.png']);

%%%%%%%%%%%%%% dim. 20
hf = figure('Position',[0,0,1000,1000]);
stairs(sort(BEST2_10(1:16)),[0:15]/15,'s-','Linewidth',1.2,'Markersize',10)
hold on
stairs(sort(QIP(1:16)),[0:15]/15,'+-','Linewidth',1.2,'Markersize',10)
stairs(sort(BARON(1:16)),[0:15]/15,'*-','Linewidth',1.2,'Markersize',10)
stairs(sort(GUROBI(1:16)),[0:15]/15,'d-','Linewidth',1.2,'Markersize',10)
stairs(sort(CPLEX(1:16)),[0:15]/15,'x-','Linewidth',1.2,'Markersize',10)
xlim([0,4])
ylim([0,1])
xlabel('Time (in seconds)')
ylabel('Fraction of problems solved')
title('Problems with n=20')
legend('B&T(Mix)','QuadprogIP','BARON','GUROBI','CPLEX','Location','southeast')
saveas(hf,['CompareSolvers_n20.png']);

%%%%%%%%%%%%%% dim. 30
hf = figure('Position',[0,0,1000,1000]);
stairs(sort(BEST2_10(17:32)),[0:15]/15,'s-','Linewidth',1.2,'Markersize',10)
hold on
stairs(sort(QIP(17:32)),[0:15]/15,'+-','Linewidth',1.2,'Markersize',10)
stairs(sort(BARON(17:32)),[0:15]/15,'*-','Linewidth',1.2,'Markersize',10)
stairs(sort(GUROBI(17:32)),[0:15]/15,'d-','Linewidth',1.2,'Markersize',10)
stairs(sort(CPLEX(17:32)),[0:15]/15,'x-','Linewidth',1.2,'Markersize',10)
xlim([0,25])
ylim([0,1])
xlabel('Time (in seconds)')
ylabel('Fraction of problems solved')
title('Problems with n=30')
legend('B&T(Mix)','QuadprogIP','BARON','GUROBI','CPLEX','Location','southeast')
saveas(hf,['CompareSolvers_n30.png']);

%%%%%%%%%%%%%% dim. 40
hf = figure('Position',[0,0,1000,1000]);
stairs(sort(BEST2_10(33:48)),[0:15]/15,'s-','Linewidth',1.2,'Markersize',10)
hold on
stairs(sort(QIP(33:48)),[0:15]/15,'+-','Linewidth',1.2,'Markersize',10)
stairs(sort(BARON(33:48)),[0:15]/15,'*-','Linewidth',1.2,'Markersize',10)
stairs(sort(GUROBI(33:48)),[0:15]/15,'d-','Linewidth',1.2,'Markersize',10)
stairs(sort(CPLEX(33:48)),[0:15]/15,'x-','Linewidth',1.2,'Markersize',10)
xlim([0,30])
ylim([0,1])
xlabel('Time (in seconds)')
ylabel('Fraction of problems solved')
title('Problems with n=40')
legend('B&T(Mix)','QuadprogIP','BARON','GUROBI','CPLEX','Location','southeast')
saveas(hf,['CompareSolvers_n40.png']);

%%%%%%%%%%%%%% dim. 50
hf = figure('Position',[0,0,1000,1000]);
stairs(sort(BEST2_10(49:64)),[0:15]/15,'s-','Linewidth',1.2,'Markersize',10)
hold on
stairs(sort(QIP(49:64)),[0:15]/15,'+-','Linewidth',1.2,'Markersize',10)
stairs(sort(BARON(49:64)),[0:15]/15,'*-','Linewidth',1.2,'Markersize',10)
stairs(sort(GUROBI(49:64)),[0:15]/15,'d-','Linewidth',1.2,'Markersize',10)
stairs(sort(CPLEX(49:64)),[0:15]/15,'x-','Linewidth',1.2,'Markersize',10)
xlim([0,30])
ylim([0,1])
xlabel('Time (in seconds)')
ylabel('Fraction of problems solved')
title('Problems with n=50')
legend('B&T(Mix)','QuadprogIP','BARON','GUROBI','CPLEX','Location','southeast')
saveas(hf,['CompareSolvers_n50.png']);
