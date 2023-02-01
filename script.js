
const MAX_ITERATIONS = 50;

function randomBetween(min, max) {
  return Math.floor(
    Math.random() * (max - min) + min
  );
}

function calcMeanCentroid(dataSet, start, end) {
  const features = dataSet[0].length;
  const n = end - start;
  let mean = [];
  for (let i = 0; i < features; i++) {
    mean.push(0);
  }
  for (let i = start; i < end; i++) {
    for (let j = 0; j < features; j++) {
      mean[j] = mean[j] + dataSet[i][j] / n;
    }
  }
  return mean;
}

function getRandomCentroidsNaiveSharding(dataset, k) {
  // implementation of a variation of naive sharding centroid initialization method
  // (not using sums or sorting, just dividing into k shards and calc mean)
  // https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
  const numSamples = dataset.length;
  // Divide dataset into k shards:
  const step = Math.floor(numSamples / k);
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const start = step * i;
    let end = step * (i + 1);
    if (i + 1 === k) {
      end = numSamples;
    }
    centroids.push(calcMeanCentroid(dataset, start, end));
  }
  return centroids;
}

function getRandomCentroids(dataset, k) {
  // selects random points as centroids from the dataset
  const numSamples = dataset.length;
  const centroidsIndex = [];
  let index;
  while (centroidsIndex.length < k) {
    index = randomBetween(0, numSamples);
    if (centroidsIndex.indexOf(index) === -1) {
      centroidsIndex.push(index);
    }
  }
  const centroids = [];
  for (let i = 0; i < centroidsIndex.length; i++) {
    const centroid = [...dataset[centroidsIndex[i]]];
    centroids.push(centroid);
  }
  return centroids;
}

function compareCentroids(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function shouldStop(oldCentroids, centroids, iterations) {
  if (iterations > MAX_ITERATIONS) {
    return true;
  }
  if (!oldCentroids || !oldCentroids.length) {
    return false;
  }
  let sameCount = true;
  for (let i = 0; i < centroids.length; i++) {
    if (!compareCentroids(centroids[i], oldCentroids[i])) {
      sameCount = false;
    }
  }
  return sameCount;
}

// Calculate Squared Euclidean Distance
function getDistanceSQ(a, b) {
  const diffs = [];
  for (let i = 0; i < a.length; i++) {
    diffs.push(a[i] - b[i]);
  }
  return diffs.reduce((r, e) => (r + (e * e)), 0);
}

// Returns a label for each piece of data in the dataset. 
function getLabels(dataSet, centroids) {
  // prep data structure:
  const labels = {};
  for (let c = 0; c < centroids.length; c++) {
    labels[c] = {
      points: [],
      centroid: centroids[c],
      pointIndices: []
    };
  }
  // For each element in the dataset, choose the closest centroid. 
  // Make that centroid the element's label.
  for (let i = 0; i < dataSet.length; i++) {
    const a = dataSet[i];
    let closestCentroid, closestCentroidIndex, prevDistance;
    for (let j = 0; j < centroids.length; j++) {
      let centroid = centroids[j];
      if (j === 0) {
        closestCentroid = centroid;
        closestCentroidIndex = j;
        prevDistance = getDistanceSQ(a, closestCentroid);
      } else {
        // get distance:
        const distance = getDistanceSQ(a, centroid);
        if (distance < prevDistance) {
          prevDistance = distance;
          closestCentroid = centroid;
          closestCentroidIndex = j;
        }
      }
    }
    // add point to centroid labels:
    labels[closestCentroidIndex].points.push(a);
    labels[closestCentroidIndex].pointIndices.push(i)
  }
  return labels;
}

function getPointsMean(pointList) {
  const totalPoints = pointList.length;
  const means = [];
  for (let j = 0; j < pointList[0].length; j++) {
    means.push(0);
  }
  for (let i = 0; i < pointList.length; i++) {
    const point = pointList[i];
    for (let j = 0; j < point.length; j++) {
      const val = point[j];
      means[j] = means[j] + val / totalPoints;
    }
  }
  return means;
}

function recalculateCentroids(dataSet, labels, k) {
  // Each centroid is the geometric mean of the points that
  // have that centroid's label. Important: If a centroid is empty (no points have
  // that centroid's label) you should randomly re-initialize it.
  let newCentroid;
  const newCentroidList = [];
  for (const k in labels) {
    const centroidGroup = labels[k];
    if (centroidGroup.points.length > 0) {
      // find mean:
      newCentroid = getPointsMean(centroidGroup.points);
    } else {
      // get new random centroid
      newCentroid = getRandomCentroids(dataSet, 1)[0];
    }
    newCentroidList.push(newCentroid);
  }
  return newCentroidList;
}

function kmeans(dataset, k, useNaiveSharding = true) {
  if (dataset.length && dataset[0].length && dataset.length > k) {
    // Initialize book keeping variables
    let iterations = 0;
    let oldCentroids, labels, centroids;

    // Initialize centroids randomly
    if (useNaiveSharding) {
      centroids = getRandomCentroidsNaiveSharding(dataset, k);
    } else {
      centroids = getRandomCentroids(dataset, k);
    }

    // Run the main k-means algorithm
    while (!shouldStop(oldCentroids, centroids, iterations)) {
      // Save old centroids for convergence test.
      oldCentroids = [...centroids];
      iterations++;

      // Assign labels to each datapoint based on centroids
      labels = getLabels(dataset, centroids);
      centroids = recalculateCentroids(dataset, labels, k);
    }

    const clusters = [];
    for (let i = 0; i < k; i++) {
      clusters.push(labels[i]);
    }
    const results = {
      clusters: clusters,
      centroids: centroids,
      iterations: iterations,
      converged: iterations <= MAX_ITERATIONS,
    };
    return results;
  } else {
    throw new Error('Invalid dataset');
  }
}


// console.log("testing")


const margin = {left: 10, top: 10, bottom: 10, right: 10}
const width = 700 - margin.left - margin.right
const height = 800 - margin.top - margin.bottom

// let testData = [[1,1,1],
//                 [1,2,1],
//                 [-1,-1,-1],
//                 [-1,-1,-1.5],
//                 [-1,-1,-1.5]]
// console.log(kmeans(testData,2))
function sortArrayAscending(arr){
  vals = arr
  vals.sort(function(a, b){return a - b});
}
function percentRankAscending(arr, v) {
  if (typeof v !== 'number') throw new TypeError('v must be a number');
  for (var i = 0, l = arr.length; i < l; i++) {
      if (v <= arr[i]) {
          while (i < l && v === arr[i]) i++;
          if (i === 0) return 0;
          if (v !== arr[i-1]) {
              i += (v - arr[i-1]) / (arr[i] - arr[i-1]);
          }
          return i / l;
      }
  }
  return 1;
}
let dataset
let popScale
let simulation,nodes,nodeTexts
let projectionStretchY ,projectionMargin ,projection 
let stepSize,step,colorX,colorY
// let testButton,testToggle,testCheckBox,testCheckBox2,closeButton
let checkBox1,checkBox2,checkBox3,checkBox4,checkBox5,checkBox6,checkBox7,checkBox8,checkBox9,checkBox10
// let formatValue = d3.format(".2s");
function formatValue(val,ft){
  let formatter = d3.format(".2s")
  // console.log(val,ft)
  if (ft==='Population'){
      return formatter(val).replace(/G/,"B")
  }
  if (ft==='Area'){
      return formatter(val).replace(/G/,"B")
  }
  if (ft==='GINI index'){
      return formatter(val).replace(/G/,"B")
  }
  if (ft==='Happy Planet Index'){
      return formatter(val).replace(/G/,"B")
  }
  if (ft==='Human Development Index'){
      return  d3.format(".2f")(val)
  }
  if (ft==='Sustainable Economic Ddevelopment Assessment (SEDA)'){
      return formatter(val).replace(/G/,"B")
  }
  if (ft==='GDP ($ USD billions PPP)'){
      return "$ "+ formatter(val).replace(/G/,"B")
  }
  if (ft==='GDP per capita in $ (PPP)'){
      return "$ "+ formatter(val).replace(/G/,"B")
  }
  if (ft==='Health Expenditure (% of GDP)'){
      return d3.format(".2f")(val)+"%"
  }
  if (ft==='Education Expenditure (% of GDP)'){
      return d3.format(".2f")(val)+"%"
  }
  if (ft==='Military Spending (% of GDP)'){
      return d3.format(".2f")(val)+"%"
  }
  if (ft==='Government Effectiveness'){
      return d3.format(".2f")(val)+"%"
  }
    
    

}
  

let foci = [
    {x:width/2,y:height/7},
    {x:width/2,y:height/7*2},
    {x:width/2,y:height/7*3},
    {x:width/2,y:height/7*4},
    {x:width/2,y:height/7*5},
    {x:width/2,y:height/7*6},
]
let allFeatures, clusterFeats
let mouseovery,mouseoverx
allFeatures = ['Population',
'Area',
'GINI index',
'Happy Planet Index',
'Human Development Index',
'Sustainable Economic Ddevelopment Assessment (SEDA)',
'GDP ($ USD billions PPP)',
'GDP per capita in $ (PPP)',
'Health Expenditure (% of GDP)',
'Education Expenditure (% of GDP)',
'Government Effectiveness',
'Military Spending (% of GDP)',]
shortenFeatsMapping = {'Population':'Population',
'Area':'Area',
'GINI index':'GINI index',
'Happy Planet Index':'Happy Planet Index',
'Human Dev Index':'Human Development Index',
'SEDA':'Sustainable Economic Ddevelopment Assessment (SEDA)',
'GDP ($B)':'GDP ($ USD billions PPP)',
'GDP per capita ':'GDP per capita in $ (PPP)',
'Health Expense':'Health Expenditure (% of GDP)',
'Education Expense':'Education Expenditure (% of GDP)',
'Gov Effectiveness':'Government Effectiveness',
'Military Spending ':'Military Spending (% of GDP)',}
shortenFeatsReverse = {'Population':'Population',
'Area':'Area',
'GINI index':'GINI index',
'Happy Planet Index':'Happy Planet Index',
'Human Development Index':'Human Dev Index',
'Sustainable Economic Ddevelopment Assessment (SEDA)':'SEDA',
'GDP ($ USD billions PPP)':'GDP ($B)',
'GDP per capita in $ (PPP)':'GDP per capita ',
'Health Expenditure (% of GDP)':'Health Expense',
'Education Expenditure (% of GDP)':'Education Expense',
'Government Effectiveness':'Gov Effectiveness',
'Military Spending (% of GDP)':'Military Spending ',}
clusterFeats = []

checkBox1 = d3.select("#o1")
checkBox1.on("click",()=>{
    if (d3.select("#o1").property("checked")){
    clusterFeats.push(allFeatures[0])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[0])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox2 = d3.select("#o2")
checkBox2.on("click",()=>{
    if (d3.select("#o2").property("checked")){
      clusterFeats.push(allFeatures[1])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[1])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    console.log(clusterFeats)
    })
checkBox3 = d3.select("#o3")
checkBox3.on("click",()=>{
    if (d3.select("#o3").property("checked")){
    clusterFeats.push(allFeatures[2])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[2])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox4 = d3.select("#o4")
checkBox4.on("click",()=>{
    if (d3.select("#o4").property("checked")){
    clusterFeats.push(allFeatures[3])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[3])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox4_ = d3.select("#o4_")
checkBox4_.on("click",()=>{
    if (d3.select("#o4_").property("checked")){
    clusterFeats.push(allFeatures[4])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[4])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox5 = d3.select("#o5")
checkBox5.on("click",()=>{
    if (d3.select("#o5").property("checked")){
    clusterFeats.push(allFeatures[5])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[5])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox6 = d3.select("#o6")
checkBox6.on("click",()=>{
    if (d3.select("#o6").property("checked")){
    clusterFeats.push(allFeatures[6])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[6])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox7 = d3.select("#o7")
checkBox7.on("click",()=>{
    if (d3.select("#o7").property("checked")){
    clusterFeats.push(allFeatures[7])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[7])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox8 = d3.select("#o8")
checkBox8.on("click",()=>{
    if (d3.select("#o8").property("checked")){
    clusterFeats.push(allFeatures[8])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[8])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox9 = d3.select("#o9")
checkBox9.on("click",()=>{
    if (d3.select("#o9").property("checked")){
    clusterFeats.push(allFeatures[9])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[9])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })
checkBox9_ = d3.select("#o9_")
checkBox9_.on("click",()=>{
    if (d3.select("#o9_").property("checked")){
    clusterFeats.push(allFeatures[10])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[10])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })    
checkBox10 = d3.select("#o10")
checkBox10.on("click",()=>{
    if (d3.select("#o10").property("checked")){
    clusterFeats.push(allFeatures[11])
    draw1Alt()
    }else{
    clusterFeats = clusterFeats.filter((d)=>d!=allFeatures[11])
    }
    if (clusterFeats.length===0){reInit()}else{draw1Alt()}
    })

var pathG = d3.geoPath();
var projectionG = d3.geoMercator()
  .scale(35)
  .center([0,0])
  .translate([120, 120]);

// Data and color scale
var dataG = d3.map();

let datajson

d3.csv("data_processed_v2.csv").then(function(data) {
    
    dataset = data
    // console.log(dataset[155]);
    popScale= d3.scaleSqrt()
                .domain([d3.min(dataset.map((d)=>parseInt(d.Area))),
                    d3.max(dataset.map((d)=>parseInt(d.Area)))])
                .range([6,12])
    // countryColor = d3.scaleOrdinal(d3.schemeTableau10)
    // .domain(dataset.map((d)=>d.Country).values());
    

    projectionStretchY = 0.25
    projectionMargin = 12
    projection = d3.geoEquirectangular()
            .scale((width / 2 - projectionMargin) / Math.PI)
            .translate([width / 2, height* (1 - projectionStretchY )/ 2]);
    // stepSize = 12
    // size = 174
    // newdataset= []
    // dataset = dataset.map((d)=>{
    //   var node = d
    //   var p = projection([d.longitude, d.latitude])
    //   node["longitudeProj"] = p[0]
    //   node["latitudeProj"] = p[1]
    //   newdataset.push(node)
    // })
    // dataset = newdataset
    // colorX = d3.scaleLinear()
    //   .domain([d3.min(dataset.map((d)=>d.longitudeProj)), d3.max(dataset.map((d)=>d.longitudeProj))])
    //   .range(['#FFF6EA', '#FD8B3C']);
    // colorY = d3.scaleLinear()
    //   .domain([d3.min(dataset.map((d)=>d.latitudeProj)), d3.max(dataset.map((d)=>d.latitudeProj))])
    //   .range(['#FFF6EA', '#68A4CC']);
      
    countryColor = d3.scaleOrdinal(d3.schemeTableau10)
      .domain(dataset.map((d)=>d['ISO Country code']).values());
    
      fetch('./world.geojson')
      .then((response) => response.json())
      .then((json) =>{ datajson = json
        


        setTimeout(drawInit(), 100)
      });
    


    
    
    // console.log("done")
    // draw1();
  });

function drawInit(){
  // function countryColor(d){
  //   var color = d3.scaleLinear()
  //       .domain([-1,1])
  //       .range([colorX(d.longitudeProj), colorY(d.latitudeProj)])
  //       .interpolate(d3.interpolateLab);

  //     var strength = (colorY(d.latitudeProj) - colorX(d.longitudeProj)) / (size-1);
  //     // console.log(colorY(d.latitudeProj) , colorX(d.longitudeProj))
  //     // console.log(d3.interpolateLab(colorY(d.latitudeProj),colorX(d.longitudeProj))(0.5))
  //     return d3.interpolateRgb(colorY(d.latitudeProj),colorX(d.longitudeProj))(0.5)
  // }
  // console.log(datajson)
  // var legendsvg = d3.select(".info")
  // .append("svg")
  //   .attr("width", 200)
  //   .attr("height", 40)
  // var valuesToShow = [d3.min(dataset.map((d)=>parseInt(d.Area))),
  // (d3.min(dataset.map((d)=>parseInt(d.Area)))+
  // d3.max(dataset.map((d)=>parseInt(d.Area))) )/2 ,
  // d3.max(dataset.map((d)=>parseInt(d.Area)))]
  // var yCircle = 15
  // var yLegend = 38
  // var xCircle = 45
  // var dxCircle = 30
  
  // legendsvg
  // .selectAll("legend")
  // .data(valuesToShow)
  // .enter()
  // .append("circle")
  // .attr("cx", (d,i)=>{return (i*dxCircle)+xCircle})
  // .attr("cy", function(d){ return yCircle } )
  // .attr("r", function(d){ return popScale(d) })
  // .style("fill", "grey")
  // .style('opacity',0.7)
  // .attr("stroke", "black")

  // // Add legend: labels
  // legendsvg
  // .selectAll("legend")
  // .data(valuesToShow)
  // .enter()
  // .append("text")
  // .attr('x', (d,i)=>{return (i*dxCircle)+xCircle})
  // .attr('y', yLegend )
  // .text( function(d){ return formatValue(d,"Area") } )
  // .style("font-size", 10)
  // // .attr('alignment-baseline', 'middle')
  // .style("text-anchor", "middle")

  // legendsvg
  // .selectAll("legend")
  // .data(["Area"])
  // .enter()
  //   .append("text")
  //   .attr('x',5)
  //   .attr('y',yLegend)
  //   .text((d)=>d)
  //   .style("font-size", 12);

  newdataset = []
    dataset.map((d)=>{
      d['kgroup']=0
      newdataset.push(d)
    })
    dataset=newdataset
    let svg = d3.select("#vis")
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('opacity', 1)
            .style("background", "#FFFCF6")
            .attr('class','mainVis')
    // .attr("background-color",'black')

    svg.append("g")
    .selectAll(".mainMap")
    .data(datajson.features)
    .enter()
    .append("path")
    .attr('class','mainMap')
      // draw each country
      .attr("d", d3.geoPath()
        .projection(projectionG)
      )
      .attr('opacity',0.7)
      // set the color of each country
      .attr("fill", function (d) {
        return countryColor(d.id)
      });
    
    simulation = d3.forceSimulation(dataset)

     // Define each tick of simulation
    simulation.on('tick', () => {
        nodes
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
        nodeTexts
            .attr('x', d => d.x)
            .attr('y', d => d.y)
    })

    // Stop the simulation until later
    simulation.stop()
    nodes = svg
        .selectAll('circle')
        .data(dataset)
        .enter()
        .append('circle')
            .attr('fill', (d)=>countryColor(d['ISO Country code']))
            .attr('opacity', 0.7)
            .attr('cx', (d, i) => projection([d.longitude, d.latitude])[0])
            .attr('cy', (d, i) => projection([d.longitude, d.latitude])[1]* (1 + projectionStretchY))
            .attr('id',(d)=>d['code'])
            // .attr('r', (d)=>1)
            // .transition().delay(2000)
            .attr('r', (d)=>popScale(parseInt(d.Area)))

    nodeTexts = svg
            .selectAll('text')
            .data(dataset)
            .enter()
            .append('text')
                .text((d)=>d['code'])
                .attr("font-weight", 44)
                .attr("font-size", 7)
                .attr('fill','#352F44')
                .attr('font-family','Inter')
                .style("text-anchor", "middle")
                .attr('transform','translate(0,2)')
                // .attr('x', (d, i) => projection([d.longitude, d.latitude])[0])
                // .attr('y', (d, i) => projection([d.longitude, d.latitude])[1]* (1 + projectionStretchY))   
    
    simulation  
            .force('charge', d3.forceManyBody().strength([0.1]))
            .force('forceX', d3.forceX(d =>projection([d.longitude, d.latitude])[0]).strength([0.05]))
            .force('forceY', d3.forceY(d => projection([d.longitude, d.latitude])[1]* (1 + projectionStretchY) ).strength(0.7))
            .force('collide', d3.forceCollide(d => popScale(parseInt(d.Area)))).tick(100)
            .alphaDecay([0.05])
    
        //Reheat simulation and restart
        simulation.alpha(1).restart()    

        svg.selectAll('circle')
        .on('mouseover', mouseOver)
        .on('mouseout', mouseOut)
        svg.selectAll('text')
        .on('mouseover', mouseOverTex)
        .on('mouseout', mouseOutTex)
        .on('click',openModal)
        svg.selectAll('circle')
          .on('click',openModal)

        function openModal(d,i){
          console.log(d)
          let modal = d3.select("#myModal")
          modal.style('display','block')
          d3.select('#modalHeading').text(d.Country)
          
          d3.select(".close").on('click',(d_)=>{
            // console.log('close')
            d3.select(".modalLeftContainer").selectAll("svg").remove().exit()
            modal.style('display','none')
            d3.select(".modalRightContainer").selectAll("table").remove().exit()
          })
          let modalwidth = 700
          let modalheight = 500
          let axisAllowanceX = 97
          let axisAllowanceY = 20
          let xaxisPointStart = 37
          let yaxisPointStart = 20
          let xLastDistanceFromEnd = 370
          let Xpointextend = modalwidth-xLastDistanceFromEnd-axisAllowanceX
          // d3.select('#modalContent1').append('p').append('text').text(d.Country)
          let modalsvg = d3.select(".modalLeftContainer").append('svg').attr("class","modalImgStyle")
                        .attr('width',modalwidth).attr('height',modalheight)
          let leftScale = d3.scaleBand().rangeRound([0, modalheight-axisAllowanceY])
                        leftScale.domain(Object.keys(shortenFeatsMapping))
          let bottomScale = d3.scaleBand().rangeRound([0, modalwidth-axisAllowanceX-xLastDistanceFromEnd])
                        bottomScale.domain(['low','','high'])
          modalsvg.append('g').attr("class","leftAAxis")
                        .attr('font-size',8)
                          .attr('transform',`translate(${axisAllowanceX},0)`)
                          .call(d3.axisLeft(leftScale))
          modalsvg.append('g').attr("class","bottomAAxis")
                          .attr('font-size',8)
                            .attr('transform',`translate(${axisAllowanceX},${modalheight - axisAllowanceY})`)
                            .call(d3.axisBottom(bottomScale));
          

          let gdataset = dataset.filter((d_)=>d_.kgroup==d.kgroup)
          // console.log(gdataset.length,clusterFeats.length)

          // modalsvg.append("foreignObject")
          //                   .attr("width", 280)
          //                   .attr("height", 200)
          //                   .attr('z-index',1000)
          //                   .attr('x',modalwidth-xLastDistanceFromEnd+20)
          //                   .attr('y',100)
          //                 .append("xhtml:body")
          //                   .style("font", "14px 'Infer'")
          //                   .html("<div class='isightclass'><p>Insights</p><p>Insights</p><p>Insights</p></div>");
          // generating insights
          let insight1;
          if (clusterFeats.length===0){
            insight1 = "There are "+gdataset.length+" countries compared here."
          }else{
            insight1 = "This cluster has "+gdataset.length+" countries with similar "
            clusterFeats.map((d,i)=>{
              if (i<clusterFeats.length-1){
                insight1 = insight1+shortenFeatsReverse[d]+", "
              }else{
                insight1 = insight1 + shortenFeatsReverse[d]+"."
              }
            })
          }
          console.log(insight1)
          let insight2;
          
          let query_country = dataset.filter((d_)=>d_.code==d.code)[0]
          let query_vectors_base = []
          clusterFeats.map((e)=>{
                  query_vectors_base.push(query_country["sc_"+e])
                  })
          let tempgdataset = []
          gdataset.map((v)=>{
              let cnt_vec = []
              old_node = v
              clusterFeats.map((e)=>{
                  cnt_vec.push(v["sc_"+e])
                  })
              old_node['qdistance'] = 1/(getDistanceSQ(cnt_vec,query_vectors_base)+1)
              tempgdataset.push(old_node) 
              })
          function compare( a, b ) {
            if ( a.qdistance > b.qdistance ){
              return -1;
            }
            if ( a.qdistance < b.qdistance ){
              return 1;
            }
            return 0;
          }
          tempgdataset.sort(compare)
          tempgdataset = tempgdataset.filter((t)=>t.code!=d.code)
          console.log(d.Country)
          // console.log(tempgdataset.slice(0,3).map((t)=>t.Country))
          tempgdataset = tempgdataset.slice(0,3)

          if (clusterFeats.length===0){
            insight2 = "Please select indicators to get more cluster insights"
          }else{
            insight2 = "In this cluster "
            tempgdataset.map((v,vi)=>{
              if (vi!=tempgdataset.length-1){
                insight2 = insight2+v.Country+", "
              }else{
                if (tempgdataset.length!=1){
                insight2 = insight2+"and " +v.Country +" have similar indicators as "+d.Country+"."}
                else{
                  insight2 = insight2+"" +v.Country +" has similar indicators as "+d.Country+"."
                }
              }
            })
          }
          console.log(insight2)
          function wrap(text, width) {
            text.each(function () {
                var text = d3.select(this),
                    words = text.text().split(/\s+/).reverse(),
                    word,
                    line = [],
                    lineNumber = 0,
                    lineHeight = 1.1, // ems
                    x = text.attr("x"),
                    y = text.attr("y"),
                    dy = 0, //parseFloat(text.attr("dy")),
                    tspan = text.text(null)
                                .append("tspan")
                                .attr("x", x)
                                .attr("y", y)
                                .attr("dy", dy + "em");
                while (word = words.pop()) {
                    line.push(word);
                    tspan.text(line.join(" "));
                    if (tspan.node().getComputedTextLength() > width) {
                        line.pop();
                        tspan.text(line.join(" "));
                        line = [word];
                        tspan = text.append("tspan")
                                    .attr("x", x)
                                    .attr("y", y)
                                    .attr("dy", ++lineNumber * lineHeight + dy + "em")
                                    .text(word);
                    }
                }
            });
        }
          let insight3,insight4,top10,bottom10;
          top10=[]
          bottom10=[]
          if (clusterFeats.length===0){
            console.log("not clustered")
            
            modalsvg.append('g')
                    .append('text')
                    .text('Insights')
                    .attr('x',modalwidth-xLastDistanceFromEnd+20)
                    .attr('y',20)
                    .attr('font-size',20)
                    // .attr('fill','blue')
            modalsvg.append('g')
                    .append('text')
                    .attr('x',modalwidth-xLastDistanceFromEnd+20)
                    .attr('y',60)
                    .text(insight1)
                    .call(wrap,xLastDistanceFromEnd-20)
            modalsvg.append('g')
                    .append('text')
                    .attr('x',modalwidth-xLastDistanceFromEnd+20)
                    .attr('y',90)
                    .text(insight2)
                    .call(wrap,xLastDistanceFromEnd-20)
          }else if(gdataset.length<=4){
         
  modalsvg.append('g')
          .append('text')
          .text('Insights')
          .attr('x',modalwidth-xLastDistanceFromEnd+20)
          .attr('y',20)
                    .attr('font-size',20)
          
  modalsvg.append('g')
          .append('text')
          .attr('x',modalwidth-xLastDistanceFromEnd+20)
          .attr('y',60)
          .text("Too few countries in Cluster!!")
          .call(wrap,xLastDistanceFromEnd-20)
  modalsvg.append('g')
          .append('text')
          .attr('x',modalwidth-xLastDistanceFromEnd+20)
          .attr('y',90)
          .text("Check out some other cluster to get more insights")
          .call(wrap,xLastDistanceFromEnd-20)

          }else{
            console.log("clustered")
            modalsvg.append('g')
          .append('text')
          .text('Insights')
          .attr('x',modalwidth-xLastDistanceFromEnd+20)
          .attr('y',20)
                    .attr('font-size',20)
            modalsvg.append('g')
                  .append('text')
                  .attr('x',modalwidth-xLastDistanceFromEnd+20)
                  .attr('y',80)
                  .text(insight1)
                  .attr('font-size',12)
                  .call(wrap,xLastDistanceFromEnd-20)
            modalsvg.append('g')
                  .append('text')
                  .attr('x',modalwidth-xLastDistanceFromEnd+20)
                  .attr('y',150)
                  .text(insight2)
                  .attr('font-size',12)
                  .call(wrap,xLastDistanceFromEnd-20)


            allFeatures.map((ft)=>{
              let arr = gdataset.map((d_)=>parseFloat(d_[ft]))
              arr.sort(function(a, b){return a - b})
              // console.log(arr[0],d[ft])
              let rankpct = 1 - percentRankAscending(arr,parseFloat(d[ft]))
              // console.log(rankpct)
              if (rankpct<=0.1){
                top10.push(ft)
              }
              if(rankpct>=0.9){
                bottom10.push(ft)
              }
            })
            let htmlbase= `<div class='isightclass'><p class='insightHeader'>Insights</p><p class='insight1'>${insight1}</p><p class='insight2'>${insight2}</p>`
            if (top10.length!=0){
              // console.log("top10")
              // console.log(top10)
              insight3 = `${d.Country} is ranked in top 10 percentile in `
              top10.map((ft_,fti_)=>{
                if (fti_===top10.length-1){
                insight3 = insight3 + shortenFeatsReverse[ft_]+" within this cluster."}else{
                  insight3 = insight3+shortenFeatsReverse[ft_]+", "
                }
                
              })
              htmlbase = htmlbase+`<p class='insight3'>${insight3}</p>`
              modalsvg.append('g')
                  .append('text')
                  .attr('x',modalwidth-xLastDistanceFromEnd+20)
                  .attr('y',210)
                  .text(insight3)
                  .attr('font-size',12)
                  .call(wrap,xLastDistanceFromEnd-20)
                  .attr('fill',"#59A14F")
            }
            if (bottom10.length!=0){
              // console.log("bot10")
              insight4 = `${d.Country} is ranked in bottom 10 percentile in `
              bottom10.map((ft_,fti_)=>{
                if (fti_===bottom10.length-1){
                  insight4 = insight4 + shortenFeatsReverse[ft_]+"."}else{
                  insight4 = insight4+ shortenFeatsReverse[ft_]+", "
                }
              })
              htmlbase = htmlbase+`<p class='insight4'>${insight4}</p>`
            }
            
            htmlbase = htmlbase+"</div>"
            // console.log(htmlbase)
            modalsvg.append('g')
                .append('text')
                .attr('x',modalwidth-xLastDistanceFromEnd+20)
                .attr('y',270)
                .text(insight4)
                .attr('font-size',12)
                .call(wrap,xLastDistanceFromEnd-20)
                .attr('fill',"#F28E2C")
          //  modalsvg.append("foreignObject")
          //                   .attr("width", 280)
          //                   .attr("height", 400)
          //                   .attr('z-index',1000)
          //                   .attr('x',modalwidth-xLastDistanceFromEnd+20)
          //                   .attr('y',100)
          //                 .append("xhtml:body")
          //                   .style("font", "14px 'Infer'")
          //                   .html(htmlbase)

          
          // .attr('fill','blue')
          
          
          
          }
          // let insight3;
          


          
          allFeatures.map((feat,index)=>{
            // let feat = "sc_"+vfeat
          
            // .attr("width")
  
  // console.log(feat)
            
            let group1 = modalsvg.append('g')
              .attr('transform',`translate(70,${yaxisPointStart})`)
              .attr('class',"circleGroup")
            let gx = d3.scaleLinear().domain([d3.min(gdataset.map((d_)=>parseFloat(d_[feat]))),
              d3.max(gdataset.map((d_)=>parseFloat(d_[feat])))]).range([xaxisPointStart,Xpointextend])
            // let gx = d3.scaleLinear().range([20,460])
            // gx.domain(d3.extent(gdataset.map((v)=>v[feat])))
            group1.selectAll('circle')
            .data(gdataset)
            .enter()
            .append('circle')
                .attr("id",(d_)=>d_["ISO Country code"]+index)
                .attr('fill', (d_)=>countryColor(d_['ISO Country code']))
                .attr('opacity', 0.8)
                .attr('cx', (d_, i) =>0)
                .attr('cy', (d_, i) => leftScale(shortenFeatsReverse[feat]))
                .attr('r', (d_)=>0)
                .attr('stroke',(d_)=>{
                  if (d_.Country===d.Country){return '#EE6C12'}else{return 'none'}
                })
                .attr('stroke-width',1)
                .on('mouseover',(d_)=>{
                  // console.log(`<strong>${d_.Country}</strong><br>
                  // ${feat}: ${d_[feat]}`)
                  d3.select("#tooltip")
                   .style('left', (d3.event.pageX + 10)+ 'px')
                    .style('top', (d3.event.pageY - 25) + 'px')
                    .style('display', 'inline-block')
                    .style("background", "white")
                    .html(`<strong>${d_.Country}</strong><br>
                          ${feat}: ${formatValue(d_[feat],feat)}`)

                })
                .on('mouseout',(d_)=>{
                  d3.select("#tooltip")
                  //  .style('left', (d3.event.pageX + 10)+ 'px')
                  //   .style('top', (d3.event.pageY - 25) + 'px')
                    .style('display', 'none')
                    // .style("background", "#FFFCF6")
                    // .html(`<strong>${d_.Country}</strong><br>
                    //       ${feat}: ${d_[feat]}`)

                })
                .transition().delay(400)
                .attr('r', (d_)=>3)
                .attr('cx', (d_, i) => gx(d_[feat]))
                .transition().delay(400)
                .attr('stroke-width',12)
          })

          
          let rightBlock = d3.select(".modalRightContainer")
                            .append("table")
                            .style("border-collapse", "collapse")
                            .style("border", "2px #F1D46D solid");
            rightBlock.append("thead").append("tr")
                            .selectAll("th")
                            .data(["Index","Value","Rank %"])
                            .enter().append("th")
                            .text(function(d_) { return d_; })
                            .style("border", "3px #F1D46D solid")
                            .style("padding", "5px")
                            .style("background-color", "#F1D46D")
                            .style("font-weight", "8")
                            // .style("text-transform", "uppercase");

          
          
          rightBlock.append("tbody")
                            .selectAll("tr").data(Object.keys(shortenFeatsMapping))
                            .enter().append("tr")
                            .selectAll("td")
                            .data(function(ft){
                              let arr = gdataset.map((d_)=>parseFloat(d_[shortenFeatsMapping[ft]]))
                              arr.sort(function(a, b){return a - b})
                              // console.log(arr[0],d[shortenFeatsMapping[ft]])
                              let rankpct = percentRankAscending(arr,parseFloat(d[shortenFeatsMapping[ft]]))
                              return [ft, 
                              formatValue(d[shortenFeatsMapping[ft]],shortenFeatsMapping[ft]),
                              d3.format(".2f")((1 - rankpct)*100)+"%"
                            ];})
                            .enter().append("td")
                            .style("border", "1px #F1D46D white")
                            .style("padding", "5px")
                            .on("mouseover", function(){
                            d3.select(this).style("background-color", "#F1D46D");
                          })
                            .on("mouseout", function(){
                            d3.select(this).style("background-color", "white");
                          })
                            .text(function(d_){return d_;})
                            .style("font-size", "12px");
                


          // annotation arrows
          let fromCoords,toCoords;
          fromCoords=[]
          toCoords=[]
          
          top10.map((ft)=>{
            let gx = d3.scaleLinear().domain([d3.min(gdataset.map((d_)=>parseFloat(d_[ft]))),
              d3.max(gdataset.map((d_)=>parseFloat(d_[ft])))]).range([xaxisPointStart,Xpointextend])
            let cx = gx(d[ft])
            let cy = leftScale(shortenFeatsReverse[ft])
            fromCoords.push([cx+70,cy+yaxisPointStart])
            // toCoords.push([modalwidth-xLastDistanceFromEnd+20,100])
            toCoords.push([modalwidth-xLastDistanceFromEnd+20,210])
            })


          //   })
          console.log(fromCoords)
          console.log(toCoords)
          
          var line = d3.line()
          .x(d => d.x)
          .y(d => d.y)
          .curve(d3.curveBasis);
            ;
          for (var i=0; i < fromCoords.length; i++) {
            let d1 = [{'x':fromCoords[i][0],"y":fromCoords[i][1]},{'x':fromCoords[i][0],"y":fromCoords[i][1]}]
            let d2 = [{'x':fromCoords[i][0],"y":fromCoords[i][1]},{'x':toCoords[i][0],"y":toCoords[i][1]}]
            modalsvg.append("path")
            .attr("class", "plot")
            // .datum(d2)
            .attr("d", line(d2))
            .attr("fill", "none")
            .attr("stroke", "none")
            .transition().delay(1500)
            .attr("stroke", "#59A14F")
            
                
                ;}
          // annotation 3
          fromCoords=[]
          toCoords=[]
          bottom10.map((ft)=>{
            let gx = d3.scaleLinear().domain([d3.min(gdataset.map((d_)=>parseFloat(d_[ft]))),
              d3.max(gdataset.map((d_)=>parseFloat(d_[ft])))]).range([xaxisPointStart,Xpointextend])
            let cx = gx(d[ft])
            let cy = leftScale(shortenFeatsReverse[ft])
            fromCoords.push([cx+70,cy+yaxisPointStart])
            // toCoords.push([modalwidth-xLastDistanceFromEnd+20,100])
            toCoords.push([modalwidth-xLastDistanceFromEnd+20,270])
            })


          //   })
          console.log(fromCoords)
          console.log(toCoords)
          
          var line = d3.line()
          .x(d => d.x)
          .y(d => d.y)
          .curve(d3.curveBasis);
            ;
          for (var i=0; i < fromCoords.length; i++) {
            let d1 = [{'x':fromCoords[i][0],"y":fromCoords[i][1]},{'x':fromCoords[i][0],"y":fromCoords[i][1]}]
            let d2 = [{'x':fromCoords[i][0],"y":fromCoords[i][1]},{'x':toCoords[i][0],"y":toCoords[i][1]}]
            modalsvg.append("path")
            .attr("class", "plot")
            // .datum(d2)
            .attr("d", line(d2))
            .attr("fill", "none")
            .attr("stroke", "none")
            .transition().delay(1500)
            .attr("stroke", "#F28E2C")
            
               
                ;}
          
           




        }

        
        
        function mouseOverTex(d, i){

          // console.log('hi')
          d3.select(this)
              .transition('mouseover').duration(100)
              .attr('opacity', 1)
              // .attr('stroke-width', 5)
              // .attr('stroke', 'black')
              d3.selectAll("#"+d.code)
              .transition('mouseover').duration(100)
              .attr('opacity', 1)
              .attr('stroke-width', 5)
              .attr('stroke', '#ffa500')
              d3.select("#tooltipMain")
              .style('left', (d3.event.pageX + 10)+ 'px')
              .style('top', (d3.event.pageY - 25) + 'px')
              .style('display', 'inline-block')
              .style("background", "#FFFCF6")
              .html(`${d.Country}`)
             
            }
        function mouseOutTex(d, i){
        
              // d3.select('#tooltip').select("circle").remove()
      
              d3.select(this)
                  .transition('mouseout').duration(100)
                  .attr('opacity', 0.8)
                  // .attr('stroke-width', 0)
              let highlightSelection = d3.select("#myList1").property('value')
              if (highlightSelection!=d.Country){
                d3.selectAll("#"+d.code)
                .transition('mouseout').duration(100)
                .attr('opacity', 0.7)
                .attr('stroke-width', 0)
              }else{
                d3.selectAll("#"+d.code)
                .transition('mouseout').duration(100)
                .attr('opacity', 0.7)
                .attr('opacity', 1)
                .attr('stroke-width', 3)
                .attr('stroke', 'black')
              }
                  
                  d3.select("#tooltipMain")
                  .style('display', 'none');
          }

    function mouseOver(d, i){

        // console.log('hi')
        d3.select(this)
            .transition('mouseover').duration(100)
            .attr('opacity', 1)
            .attr('stroke-width', 5)
            .attr('stroke', '#ffa500')
        
          d3.select("#tooltipMain")
          .style('left', (d3.event.pageX + 10)+ 'px')
          .style('top', (d3.event.pageY - 25) + 'px')
          .style('display', 'inline-block')
          .style("background", "#FFFCF6")
          .html(`${d.Country}`)



          }
    
    function mouseOut(d, i){
        
        // d3.select('#tooltip').select("circle").remove()
        let highlightSelection = d3.select("#myList1").property('value')
        if (highlightSelection!=d.Country){
          d3.select(this)
            .transition('mouseout').duration(100)
            .attr('opacity', 0.7)
            .attr('stroke-width', 0)
        }else{
          d3.select(this)
          .transition('mouseout').duration(100)
          .attr('opacity', 0.7)
          .attr('opacity', 1)
          .attr('stroke-width', 3)
          .attr('stroke', 'black')
        }
        // d3.select(this)
        //     .transition('mouseout').duration(100)
        //     .attr('opacity', 0.7)
        //     .attr('stroke-width', 0)
            // d3.select("#tooltipMain").remove().exit();
            d3.select("#tooltipMain")
            .style('display', 'none');
            // d3.select("#tooltipMain").remove().exit();



    }
    var highlight_list = ["Select Country"]
    dataset.map((d)=>{highlight_list.push(d.Country)})
    d3.select("#myList1").selectAll("option")
      .data(highlight_list)
      .enter()
      .append('option')
      .text((d)=>d)
      .attr('value',(d)=>(d));
      d3.select("#myList1").on('change',()=>{
        // console.log(d3.select("#myList1").property('value'))
        let query = d3.select("#myList1").property('value')
        if (query!="Select Country"){
        query = dataset.filter((d)=>d.Country==query)[0]['code']
        // console.log(query)
        document.querySelector("#"+query).scrollIntoView({
          behavior: 'smooth'
      });
        d3.selectAll("#"+query).transition().duration(500)
        .attr('opacity', 1)
        .attr('stroke-width', 2)
        .attr('stroke', 'black')
        .transition().duration(1000)
        .attr('opacity', 0.5)
        .attr('stroke-width', 15)
        .attr('stroke', 'red')
        .transition().duration(1000)
        .attr('opacity', 1)
        .attr('stroke-width', 3)
        .attr('stroke', 'black')
        .transition().duration(1000)
        .attr('opacity', 1)
        .attr('stroke-width', 15)
        .attr('stroke', 'red')
        .transition().duration(1000)
        .attr('opacity', 1)
        .attr('stroke-width', 3)
        .attr('stroke', 'black')}else{
          d3.selectAll('circle')
          .transition('mouseout').duration(100)
                .attr('opacity', 0.7)
                .attr('stroke-width', 0)
        }
   
      })
      
      
      var valuesToShow = [d3.min(dataset.map((d)=>parseInt(d.Area))),
      (d3.min(dataset.map((d)=>parseInt(d.Area)))+
      d3.max(dataset.map((d)=>parseInt(d.Area))) )/2 ,
      d3.max(dataset.map((d)=>parseInt(d.Area)))]
      var yCircle = 15
      var yLegend = 38
      var xCircle = 45
      var dxCircle = 30
      var legendsvg = svg
      .append("g")
       .attr('transform', `translate(${width/2-xCircle},10)`)
      legendsvg
      .selectAll("legend")
      .data(valuesToShow)
      .enter()
      .append("circle")
      .attr("cx", (d,i)=>{return (i*dxCircle)+xCircle})
      .attr("cy", function(d){ return yCircle } )
      .attr("r", function(d){ return popScale(d) })
      .style("fill", "grey")
      .style('opacity',0.7)
      .attr("stroke", "black")
    
      // Add legend: labels
      legendsvg
      .selectAll("legend")
      .data(valuesToShow)
      .enter()
      .append("text")
      .attr('x', (d,i)=>{return (i*dxCircle)+xCircle})
      .attr('y', yLegend )
      .text( function(d){ return formatValue(d,"Area") } )
      .style("font-size", 10)
      // .attr('alignment-baseline', 'middle')
      .style("text-anchor", "middle")
    
      legendsvg
      .selectAll("legend")
      .data(["Area"])
      .enter()
        .append("text")
        .attr('x',5)
        .attr('y',yLegend)
        .text((d)=>d)
        .style("font-size", 12);
    




}
function getReportingMetric(){
  return clusterFeats[0]
}
function draw1Alt(){
  d3.selectAll('.mainMap').remove().exit()
  let dataToCluster   = dataset.map((d)=>{
      let arr = []
      clusterFeats.map((col)=>{
        arr.push(d["sc_"+col])
      })
      
      return arr
  })
  let kmeansResult = kmeans(dataToCluster,6)
  
  for (var key in kmeansResult.clusters){
      // console.log( key, kmeansResult.clusters[key].pointIndices );
      for (let i = 0; i < kmeansResult.clusters[key].pointIndices.length; i++) {
          dataset[kmeansResult.clusters[key].pointIndices[i]]["kgroup"] = parseInt(key)
        }
    }
  
  var reportingMetric = getReportingMetric()
  let sortable = []
  d3.set(dataset.map((d)=>d.kgroup)).values().map((d,i)=>{
    // console.log(d,i)
    // console.log(d,dataset.filter((d_)=>d_.kgroup===d).length);
    sortable.push([d,d3.mean(dataset.filter((d_)=>d_.kgroup==d).map((d_)=>d_[reportingMetric]))])
  }
  )
  // console.log(sortable);
  sortable.sort(function(a, b) {
    return b[1] - a[1];
    });
    // console.log(sortable);
  newMapping = {}
  sortable.map((d,i)=>
  {
    newMapping[d[0]]=i
  })
  // console.log("new mapping",newMapping)
  newdataset = []
  dataset.map((d)=>{
    newobject = d
    newobject["kgroup"] = newMapping[d["kgroup"]]
    newdataset.push(newobject)
  })
  dataset = newdataset
  // console.log(newdataset[0])
  // console.log(dataset)
  simulation.stop()
  simulation  
  .force('charge', d3.forceManyBody().strength([0.1]))
  .force('forceX', d3.forceX(d => foci[d.kgroup].x).strength([0.05]))
  .force('forceY', d3.forceY(d =>  foci[d.kgroup].y ).strength([0.2]))
  .force('collide', d3.forceCollide(d => popScale(d.Area) ))
  .alpha(0.7).alphaDecay([0.05])

//Reheat simulation and restart
simulation.restart()
let svg = d3.select("#vis")
  .select('svg');
  svg.selectAll(".annotations").remove().exit()
  svg.selectAll(".annotationMap").remove().exit()
  const annotations = [
    // {
    //   note: {
    //     label: "Here is the annotation label",
    //     title: "Annotation title",
    //     align: "middle",  // try right or left
    //     wrap: 200,  // try something smaller to see text split in several lines
    //     padding: 10,   // More = text lower
        
    //   },
    //   color: ["grey"],
    //   // font-family: 'Inter',
    //   x: foci[0].x+150,
    //   y: foci[0].y,
    //   dy: -20,
    //   dx: 200
    // }
     // label:  `Mean ${reportingMetric}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[reportingMetric])))}`,
    //  title: `Mean ${reportingMetric}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[reportingMetric])))}`,
  ]
  foci.map((d,i)=>{
    // annotations.push()
    clusterFeats.slice(0,4).map((ft,i_)=>{
      // annotations.push([d,d_,shortenFeatsReverse[ft],d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft]))])
      svg.append('g')
        .attr('class','annotations')
        .append('text')
        .text(`${shortenFeatsReverse[ft]}: ${formatValue(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft])),ft)}`)
        .attr("font-weight", 44)
        .attr("font-size", 0)
        .attr('fill','#352F44')
        .attr('font-family','Inter')
        .style("text-anchor", "middle")
        .attr('x', d.x+(width/2.9))
        .attr('y', d.y-height/50 + i_*20) 
        .transition().delay(i*200)
        .attr("font-size", 12)

    })
    let projectionGC = d3.geoMercator()
    .scale(25)
    .center([0,0])
    .translate([ d.x-(width/2.9), d.y-height/50 + 30]);
    svg.append("g")
    .attr('class','annotationMap')
      .selectAll(`.clusterMap${i}`)
      .data(datajson.features)
      .enter()
      .append("path")
      .attr('class',`clusterMap${i}`)
        // draw each country
        .attr("d", d3.geoPath()
          .projection(projectionGC)
        )
        // set the color of each country
        // .attr('fill','none')
        // .attr('stroke','#ffa500')
        // .transition().delay(500)
        // .attr('stroke','none')
        .attr("fill", function (v) {
          // console.log(dataset.filter((d_)=>d_["ISO Country code"]===v.id),v.id)
          if (dataset.filter((d_)=>d_["ISO Country code"]===v.id)[0].kgroup===i){
          return '#ffa500'}
          else{
            return 'grey'
          }
        })
        // .attr("opacity", 0)
        // .transition().delay(i*200)
        
        .attr("opacity", function (v) {
          // console.log(dataset.filter((d_)=>d_["ISO Country code"]===v.id),v.id)
          if (dataset.filter((d_)=>d_["ISO Country code"]===v.id)[0].kgroup===i){
          return 0.7}
          else{
            return 0.3
          }
        })
        // .attr('opacity',0)
        // .transition().delay(300)
        // .attr('opacity',1)
        ;
  })



// svg
// .append("g")
// .selectAll('.annotations')
// .data(foci)
// .enter()
// .append('text')
// .attr("class","annotations")
// .text((d,i)=>{
//   var displayText = ''
//   clusterFeats.map((ft)=>{
//     displayText = displayText + `Mean ${shortenFeatsReverse[ft]}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft])))}`
//   })
//   var ft = clusterFeats[0]
//   displayText = `Mean ${shortenFeatsReverse[ft]}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft])))}`
//   return displayText})
// .attr("font-weight", 44)
//   .attr("font-size", 0)
//   .attr('fill','#352F44')
//   .attr('font-family','Inter')
//   .style("text-anchor", "middle")
//   .attr('x', (d, i) => d.x+(width/3.9))
//   .attr('y', (d, i) => d.y) 
//   .transition().delay((d,i)=>i*200)
//   .attr("font-size", 12)
  //   foci.map((d,i)=>{
  //     d3.select("#cluster"+i)
  //     .style('left', '100px')
  //     .style('top',  '100px')
  //     .style('display', 'inline-block')
  //     .style("background", "#FFFCF6")
  //     .html(`<strong>${d}</strong><br>
  //           ${i}: ${d}`)
        

  // })
  // d3.select("#cluster"+0)
  // .style('left', '100px')
  // .style('top',  '100px')
  // .style('display', 'inline-block')
  // .style("background", "#FFFCF6")
  // .html(`<strong>asdfad</strong><br>`)
  // console.log(svg)
  // foci.map((d,i)=>{
  //   d3.select("#cluster"+i)
  //   .style('left',d.x+500+'px')
  //   .style('top',d.y+200+'px')
  //   .style('display','inline-block')
  //   .html(`<strong>asdfad</strong>`)
    
  // })
}

function reInit(){
    console.log("reInit triggerrred")
    newdataset = []
    dataset.map((d)=>{
      d['kgroup']=0
      newdataset.push(d)
    })
    dataset=newdataset
    simulation.stop()
    simulation  
    .force('charge', d3.forceManyBody().strength([0.1]))
    .force('forceX', d3.forceX(d =>projection([d.longitude, d.latitude])[0]).strength([0.04]))
    .force('forceY', d3.forceY(d => projection([d.longitude, d.latitude])[1]* (1 + projectionStretchY) ).strength(0.2))
    .force('collide', d3.forceCollide(d => popScale(parseInt(d.Area)))).tick(100)
    .alphaDecay([0.05])
    // let svg = d3.select("#vis")
   
    d3.selectAll(".annotations").remove().exit()
    d3.selectAll(".clusterMap").remove().exit()
    d3.selectAll(".annotationMap").remove().exit()
    // console.log(datajson)
    d3.select(".mainVis").append("g")
    .selectAll(".mainMap")
    .data(datajson.features)
    .enter()
    .append("path")
    .attr('class','mainMap')
      // draw each country
      .attr("d", d3.geoPath()
        .projection(projectionG)
      )
      // set the color of each country
      // .attr('fill','none')
      // .attr('stroke','#ffa500')
      // .transition().delay(500)
      .attr('stroke','none')
      
      .attr("fill", function (d) {
        return countryColor(d.id)
      })
      .attr('opacity',0)
      .transition().delay(300)
      .attr('opacity',0.7)
      ;
//Reheat simulation and restart
simulation.alpha(0.7).restart()
        
}
