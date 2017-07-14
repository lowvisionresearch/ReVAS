function GenerateRandomPattern

figure;
t = linspace(0,20*pi,10000);
x = (5*t).*cos(t-40);
y = (t.^1.5).*sin(t+30);
plot(x,y,'-k','LineWidth',2); hold on;
xlim([-255.5 255.5])
ylim([-255.5 255.5])
axis equal
box off;
axis off



figure('units','normalized','outerposition',[.3 .3 .15 .15*16/9]);
L = 512;
x = 0:64:L;
center = [L L]/2; 
w = 1;
for i=1:length(x)
    plot([0 center(1)],[x(i) center(2)],'-k','LineWidth',w); hold on;
    plot([x(i) center(1)],[0 center(2)],'-k','LineWidth',w); hold on;
    plot([L center(1)],[x(i) center(2)],'-k','LineWidth',w); hold on;
    plot([x(i) center(1)],[L center(2)],'-k','LineWidth',w); hold on;
end
plot(center(1),center(2),'ow','MarkerFaceColor','w','MarkerSize',20)
t = linspace(0,20*pi,10000);
x = (5*t).*cos(.5*t-40)+L/2;
y = (t.^1.5).*sin(.5*t+30)+L/2;
plot(x,y,'-k','LineWidth',w);

xlim([0 L])
ylim([0 L])
axis square
box off;
axis off


print('pattern.tif','-dtiff','-r300');