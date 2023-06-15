clear

% Path to main directory Rosbag3
mainPath = 'C:\Users\panh3\Desktop\Rosbag3\'
Timeinterval = 1
Mapnum = 5
rosPath = string(mainPath)+'Rosbag\map'+string(Mapnum)+'\'
d = dir(rosPath)
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
for rosread=1:length(nameFolds)
    folderName = string(nameFolds(rosread))

    folderPath = string(mainPath) + 'Rosbag\map' + string(Mapnum) + '\'+string(folderName) +'\'
    bag = ros2bagreader(folderPath)
    
    %Retrieving Odometry info
    bagOdom = select(bag,"Topic","/odom");
    Odom = readMessages(bagOdom)
    b = [Odom{:,1}]

    %Retrieving Odometry position
    Odompose = [b.pose]
    Odompose = [Odompose.pose]
    Odomposition = [Odompose.position]
    xodom = transpose([Odomposition.x])
    yodom = transpose([Odomposition.y])
    
    %Retrieving Odometry orientation
    Odomorientation = [Odompose.orientation]
    xori = transpose([Odomorientation.x])
    yori = transpose([Odomorientation.y])
    theta = atan(yori./xori)
    
    %Retrieving Odometry time
    Odomheader = [b.header]
    Odomstamp = [Odomheader.stamp]
    Odomsec = transpose([Odomstamp.sec])
    Odomnanosec = int32(transpose([Odomstamp.nanosec]))
    Odomtime = double(double(Odomsec)+double(Odomnanosec)*10^-9)

    %Retrieving LiDAR scan info
    bagScan = select(bag, "Topic","/scan")
    Scan = readMessages(bagScan)
    a = [Scan{:,1}]

    %Retrieving LiDAR scan ranges
    Ranges = transpose([a.ranges])

    %Retrieving LiDAR scan times
    Scanheader = [a.header]
    Scanstamp = [Scanheader.stamp]
    Scansec = transpose([Scanstamp.sec])
    Scannanosec = int32(transpose([Scanstamp.nanosec]))
    Scantime = double(double(Scansec)+double(Scannanosec)*10^-9)
    
    
    %retrieving amcl pose info
    bagAmcl = select(bag,"Topic","/amcl_pose")
    amcl = readMessages(bagAmcl)
    c = [amcl{:,1}]

    %retrieving amcl pose position
    amclpose = [c.pose]
    amclpose2 = [amclpose.pose]
    amclposition = [amclpose2.position]

    xamcl = transpose([amclposition.x])
    yamcl = transpose([amclposition.y])
    posamcl = [xamcl, yamcl]

    %retrieving amcl pose orientation
    amclorientation = [amclpose2.orientation]
    amclorix = transpose([amclorientation.x])
    amcloriy = transpose([amclorientation.y])
    thetaamcl = atan(amcloriy./amclorix)

    %retrieving amcl pose time
    Amclheader = [c.header]
    Amclstamp = [Amclheader.stamp]
    Amclsec = transpose([Amclstamp.sec])
    Amclnanosec = int32(transpose([Amclstamp.nanosec]))
    Amcltime = double(double(Amclsec)+double(Amclnanosec)*10^-9)

    % Gathering indexes based on closest time
    [val,idx] = min(abs(transpose(Amcltime)-Odomtime))
    xodom = xodom(idx)
    yodom = yodom(idx)
    pos = [xodom,yodom]
    theta = theta(idx)
    Ydata = [pos,theta]

    [valLS,idxLS] = min(abs(transpose(Amcltime)-Scantime))
    Ranges = Ranges(transpose(idxLS),:)
    Amcldata = [posamcl, thetaamcl]
    
end

% Retrieving Map info
bagMap = select(bag,"Topic","/map")
map = readMessages(bagMap)
Mapwidth = map{1,1}.info.width 
Mapheight = map{1,1}.info.height
Mapinfo = map{1,1}.data
Mapinfo = reshape(Mapinfo,Mapwidth, Mapheight)
Mapinfo(Mapinfo==-1)=0
% Get all rows and columns where the image is nonzero
[nonZeroRows,nonZeroColumns] = find(Mapinfo);
% Get the cropping parameters
topRow = min(nonZeroRows(:));
bottomRow = max(nonZeroRows(:));
leftColumn = min(nonZeroColumns(:));
rightColumn = max(nonZeroColumns(:));
% Extract a cropped image from the original.
croppedImage = Mapinfo(topRow:bottomRow, leftColumn:rightColumn);
croppedImage = (croppedImage-100)*-1.27

a = uint8(croppedImage)
mapnumber = "map" + string(Mapnum) + ".png"
imwrite(a, mapnumber)

mapname = 'mapdata' + string(Mapnum) + '.mat'
save(mapname,"Mapinfo","Ydata","Ranges","Amcldata","-mat")