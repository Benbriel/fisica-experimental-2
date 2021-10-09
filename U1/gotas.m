clear

%Datos
LxP=1/841.54;       %longitud por pixel (mm por pixel)
height=0.034;       % profundidad en mm
wide=0.5;           %ancho del canal en mm
Qo = 30;
Qw = 10;
Vc=0.783;           % Velocidad de la platina en mm/s
                    % 0.6 hasta 30/04, 0.876 en 30/03 y .783 en 30/01
fps=30;             %frames por segundo

X_0=1;
X_f=1000;

BIZ=200;
BDR=650;

%%%CARPETA
name=['C:\Users\RODRIGO\Documents\Uchile\6to Semestre\FE2\datos\o_' num2str(Qo,'%02i') '_d_' num2str(Qw,'%02i') '\Acq_AAA_001\'];

Fmin=327;
N=330-Fmin; 

%arreglo vacio para las imagenes
im=cell(N,1);

%guardamos y leemos las imagenes
for k=1:N
    archivo=['Cam_2304060054_000' num2str(k-1+Fmin,'%03i') '.tif'];
    imagen=imread(strcat(name,archivo));
    im{k}=imagen;
end

%%

% SELECCIONANDO MARCO DE LA GOTA
figure(1), clf
imshow(im{2});            %muestra la imagen con cuatro lineas a 100 pixeles de distancia
altura=size(im{2});
LINEV=1:1:altura(1);
lineV=1:1:altura(2);
lineI=100*ones(1,altura(2)) ;  
lineD=200*ones(1,altura(2));
lineR=300*ones(1,altura(2));
lineS=400*ones(1,altura(2));
LINE1=100*ones(altura(1),1);

%%%ENCERRANDO LA GOTA EN LA IMAGEN
ii=1;

figure(1)

IM=cell(N,1);

clear('i','IM')


for i=1:N;                   
    imshow(im{i}([BIZ:1:BDR],[X_0:1:X_f],:));
    title(i)
    drawnow;
    IM{i}=im{i}([BIZ:1:BDR],[X_0:1:X_f],:); 

end

I_f=N;

Di=1;
Df=N;

j_0=1;


%%
%%%%ANALISIS DE IMAGENES%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UM=1;
figure(1)
video=cell(N,1);
    
for i=1:I_f
    umbral=UM*graythresh(IM{i});      %decidir intensidad para blanco o negro
    BW=im2bw(IM{i},umbral);        %se decide si es negro o blanco
    BW2=imfill(~BW,'holes');           %rellena los huecos
    BW3=imclearborder(BW);             %mata todo lo que tenga contacto con el borde
    BW4=imfill(BW3,'holes');           %rellena de nuevoo para matar los huecos generados por las manchas
    BW5=bwareaopen(BW4,100);           %destruye elementos de menos de (,Pixeles)
    video{i}=BW5;
    imshow(BW5)
    title(i)                           %permite ver la numeración
    drawnow
end

i_0=2;
largo=I_f-i_0;
video1=cell(largo,1);
for i=1:largo
    video1{i}=video{i_0+i};           %VIDEO DE GOTAS CON EL ANALISIS DE IMAGEN BW
    % imshow(video1{i});
end

%%
%%%%%%%%ANALISIS DE BORDES%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Areas=cell(largo,1);           %ARREGLOS VACIOS
Centroide=cell(largo,1);
Exntr=cell(largo,1);
Perimetro=cell(largo,1);
Rdns=cell(largo,1); %redondez

for i=1:size(video1,1)
    clear BW5
    BW5=video1{i};
%     figure(1), imshow(BW5), title(i), pause % Chequeo (MLC)
    rp=regionprops(BW5,'Area','Centroid','Eccentricity','Perimeter');   %extrae propiedades geométricas de BW5
     
     if isempty(rp)==1   % si no detecta la gota llena las variables con NaN
         Areas{i}=NaN;
         Exntr{i}=NaN;
         Perimetro{i}=NaN;
         Rdns{i}=NaN;
 
         XFit(i,1)=NaN;
         XFit(i,2)=NaN;
         XFit(i,3)=NaN;
         XFit(i,4)=NaN;
         XFit(i,5)=NaN;
         XFit(i,6)=NaN;
         confi(i)=NaN;
         xr(i)=NaN;
         disp('oh, oh, error')
         continue
     else   %si las detecta calcula las propiedades geométricas
         Areas{i}=rp(1).Area;
         Exntr{i}=rp(1).Eccentricity;
         Perimetro{i}=rp(1).Perimeter;
         Rdns{i}=(rp(1).Perimeter.*rp(1).Perimeter)./(4*pi*rp(1).Area);
         
         %USA EL BORDE EDGE PARA CALCULAR LAS PROPIEDADES
%          BW6=edge(BW5); % sacar
         BW6=edge(bwmorph(BW5,'close'), 'Canny'); % MLC
         %figure(1), clf, imshow(BW6), title(i), pause % Chequeo (MLC)
         rpp=regionprops(BW6,'PixelList','Centroid');
         xm1=(rpp(1).Centroid(:,1));
         xr(i)=xm1;                           %posición relativa de la gota
         ym1=(rpp(1).Centroid(:,2));
         yr(i)=ym1; % Guardar la posicion y del centro de la gota para cada tiempo (MLC)
         pts1=rpp(1).PixelList;              %crea una lista de pixeles del borde de la gota
         x1=pts1(:,1);
         y1=pts1(:,2);
         X1=x1-xm1;                          %cambia el sistema de referencia al centro de masa
         Y1=y1-ym1;
         [Teta1,Ro1]=cart2pol(X1,Y1);        %lo escribe en coordenadas polares
         pol1=[Teta1 Ro1];
         [theta1,k1]=sort(pol1(:,1));       %ordena la lista de puntos,(considera el ángulo como criterio para ordenar
         drawnow
         R0=sqrt(rp(1).Area/pi);               %R0=Radio Promedio
         R00(i)=R0; % Guardar el radio promedio para cada tiempo (MLC)
         R1=Ro1(k1)/R0;                     %R1=Radio Adimensional

%          figure(1), clf, plot(R1.*cos(theta1),R1.*sin(theta1)), title(i), axis image, pause % Chequeo (MLC)
         
         %Se crea un ajuste polinomico de orden 5 con la gota
         cosenos3=@(x,th) x(1)+x(2)*cos(th)+x(3)*cos(2*th)+x(4)*cos(3*th)+x(5)*cos(4*th);         
         xfit1=lsqcurvefit(cosenos3,[R0,mean(X1),1,1.1,0.1],theta1,R1); % Cambiar los valores de adivinanza inicial a valores más pequeños (MLC)
         thetaajuste=-pi:pi/100:pi;
         Rajuste1=cosenos3(xfit1,thetaajuste);
         XFit(i,1)=xfit1(1);       %modo 0 (radio promedio)
         XFit(i,2)=xfit1(2);       %modo 1 (centro de masa)
         XFit(i,3)=xfit1(3);       %modo 2 (deformación elíptica)
         XFit(i,4)=xfit1(4);       %modo 3 (puntas y bala)
         XFit(i,5)=xfit1(5);       %modo 4

         confi(i)=2*R0*LxP/height; %DELTA

         %Chequear que los ajustes sean buenos (MLC)
         figure(2), clf, hold on
         plot(R1.*cos(theta1),R1.*sin(theta1))
         plot(cosenos3(xfit1,thetaajuste).*cos(thetaajuste),cosenos3(xfit1,thetaajuste).*sin(thetaajuste))
         axis image
         title(i)

     end
end

%%

Temp=21;
viscosidad=1.199;
    
Voil=Qo/(height*wide)*10^(-6);    %velocidad promedio DEL ACEITE en metros/segundo       1 microlitro es 10^9 micrometros cubicos
Ca=Voil*viscosidad/0.0414;            %Ca=viscocidad por Velocidad_OIL /tensión superficial 
xdrop=height/wide; %RELACION DE ASPECTO
delta=sprintf(' = %.2f',nanmean(confi)); %DELTA


%POSICION%


if i_0 >= j_0
    ttot=1/fps:1/fps:N/fps;        %vector de todo el tiempo             
    Xplat=ttot*Vc;        % posiciones de la platina en MM
    XR=xr;
    ttparcial=1/fps:1/fps:(i_0-j_0+size(XR,2)-1)/fps; %VECTOR PARCIAL DEL TIEMPO
    
    %posición real de la gota
    Xg=XR*LxP+Xplat(i_0-j_0: i_0-j_0+size(XR,2)-1)-LxP*X_0;             %suma la posicion relativa a la platina + la de la platina en mm
    figure(4)
    plot(Xg)
else
    % Por aquí no pasa
    ttot=1/fps:1/fps:N/fps;         
    Xplat=ttot*Vc;      
    XR=xr;
    ttparcial=1/fps:1/fps:(i_0-j_0+size(XR,2)-1)/fps;
    
    %posición de la gota
    Xplat2(1,j_0-i_0:j_0-i_0+size(Xplat,2)-1)=Xplat;
    Xg=XR*LxP+Xplat2(1,1:size(XR,2));
    Xg=Xg-LxP*X_0;
    figure(1)
    plot(Xg)
end

%calculo de la velocidad de la gota
Vg=zeros(size(Xg,2)-1,1);
for i=1:length(Vg)-1
    Vg(i)=(Xg(i+1)-Xg(i))*fps;
end

vFill = fillmissing(Vg,'constant',0); %ELIMINAMOS LOS NAN
vRemove = rmmissing(Vg);
Vmean=mean(Vg); %CALCULAMOS LA VELOCIDAD MEDIA RELATIVA A LA PLATINA
Vtotalgota=Vmean+Vc; %CALCULAMOS LA VELOCIDAD MEDIA TOTAL DE LA GOTA CON RESPECTO AL OBSERVADOR
alpha=Vtotalgota/Voil*0.001; %CALCULAMOS ALPHA COMO VGOTA/VOIL EN M/S

%PLOT DE MODOS DE FOURIER
figure(5)
Xg2=Xg(1:size(XFit,1));
plot(Xg2,XFit(:,1),'.');%modo 0 (radio promedio)
hold on
plot(Xg2,XFit(:,2),'.');%modo 1 (centro de masa)
plot(Xg2,XFit(:,3),'.');%modo 2 (deformación elíptica)
plot(Xg2,XFit(:,4),'.');%modo 3 (puntas y bala)
plot(Xg2,XFit(:,5),'.');%modo 4
plot(Xg2,0*Xg2,'k'); %LINEA CON X=0
title(['\delta' sprintf(' = %.2f',nanmean(confi)) sprintf(', Ca = %.2f',Ca)]);
xlabel('posición(mm)');
ylabel('modos');
legend('modo 0','modo 1','modo 2','modo 3','modo 4');

confir=confi;
Xgr=Xg;

%aca sacamos la velocidad de la gota


%filtro de tamaño de gota
PolAjus=polyfit(Xg,confi,3);               %se crea un polinomio de ajuste del confinamiento versus su posición
Confifit = PolAjus(4) + Xg.*PolAjus(3) + Xg.^2*PolAjus(2) + Xg.^3*PolAjus(1);

figure(3)
plot(Xg,confir,'.r') %PLOTEAMOS LA POSICION DE LA GOTA C/R AL DELTA
hold on 
plot(Xg,Confifit,'.b') %PLOTEAMOS EL AJUSTE DEL GRAFICO ANTERIOR
axis([0 max(Xg) 0 max(confi)])
hold off

Error=std(confi-Confifit);                  %se calcula el error usando la desviación estandar entre el confinamiento experimental y el ajuste de un polinomio de orden 3
Matriz=zeros(size(Xg,2),1);                  %matriz vacia que contendrá una lista de las medidas que no sirven 

for i=1:size(Xg,2)
    
         if (Confifit(i)-confi(i)).^2>= Error^2;%la matriz asigna un 1 si la distancia entre el ajuste y la medicion es mayor que el error y un 0 si desea conservarla
             Matriz(i,1)=1;
         else
             Matriz(i,1)=0;
         end
        
end
confi=confi';                         %lista de confinamiento original de respaldo
XFitr=XFit;                           %lista de modos original de respaldo

j=[find(Matriz)];
for i=1:length(j)
    confi(j(i))=NaN;
    Perimetro{j(i)}=NaN;
    Exntr{j(i)}=NaN;
    Areas{j(i)}=NaN;
    Xg=Xg';
    Xg(j(i))=NaN;
    XFit(j(i))=NaN;
end

% %%  GUARDAR DATOS
save([sprintf('Qo=%.2f',Qo) sprintf('_Qw=%.2f',Qw) sprintf('_Vc=%.2f',Vc) '_.mat'],'Qo','Qw','fps','Vc','XFit','Perimetro','Exntr','Areas','Ca','Xg','XFitr','Matriz','confi','alpha','xdrop','delta','Voil','R0','Vg','Vmean','Xg2','Fmin','N');
