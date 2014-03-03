function plotResult(x, y, xtest, ytest, xs, mu, s2, B) 
clf
hold on
set(gcf,'defaultlinelinewidth',1.5);
axis([0 2.5, -3 3]);
%plot(x,y,'.m', 'MarkerSize', 10)% data points in magenta
%plot(xtest, ytest, '-', 'Color', [0 .5 0]);
plot(xtest, ytest, '-', 'Color', 'k');
plot(xs, mu,'b') % mean predictions in blue
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
if nargin > 7
    plot(B,(min(ytest)-1)*ones(size(B)),'k+','markersize',20)
end
hold off
%axis([0 5 -0.5 1]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
end