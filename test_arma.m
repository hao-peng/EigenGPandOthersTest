for tid = 1:ns
nses(tid) = (mu(tid)-all_y(n+tid))^2;
base(tid) = (mean(all_y(1:(n+tid-1)))-y(n+tid))^2;
end
nmses_ar = sum(nses)/sum(base);
mean_nses_ar(cid) = mean(nses);
stderr_nses_ar(cid) = std(nses)/sqrt(ns);

clf
hold on
set(gcf,'defaultlinelinewidth',1.5);
axis([0 2.5, -3 3]);
plot(x,y,'.m', 'MarkerSize', 10)% data points in magenta
%plot(txtest, tytest, '-', 'Color', [0 .5 0]);
plot(txtest, tytest, '-', 'Color', 'k');
plot(x(1+n:n+ns), mu, 'b')
plot(x(1+n:n+ns),mu+2*se,'r') % plus/minus 2 std deviation predictions in red
plot(x(1+n:n+ns),mu-2*se,'r')
hold off
%axis([0 5 -0.5 1]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')