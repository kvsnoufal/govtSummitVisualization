
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


console.log("testing")


const margin = {left: 10, top: 10, bottom: 10, right: 10}
const width = 700 - margin.left - margin.right
const height = 800 - margin.top - margin.bottom

// let testData = [[1,1,1],
//                 [1,2,1],
//                 [-1,-1,-1],
//                 [-1,-1,-1.5],
//                 [-1,-1,-1.5]]
// console.log(kmeans(testData,2))

let dataset
let popScale,countryColor
let simulation,nodes,nodeTexts
// let testButton,testToggle,testCheckBox,testCheckBox2,closeButton
let checkBox1,checkBox2,checkBox3,checkBox4,checkBox5,checkBox6,checkBox7,checkBox8,checkBox9,checkBox10

let foci = [
    {x:width/4,y:height/7},
    {x:width/4,y:height/7*2},
    {x:width/4,y:height/7*3},
    {x:width/4,y:height/7*4},
    {x:width/4,y:height/7*5},
    {x:width/4,y:height/7*6},
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
'Human Development Index':'Human Development Index',
'SEDA':'Sustainable Economic Ddevelopment Assessment (SEDA)',
'GDP ':'GDP ($ USD billions PPP)',
'GDP per capita ':'GDP per capita in $ (PPP)',
'Health Expenditure':'Health Expenditure (% of GDP)',
'Education Expenditure':'Education Expenditure (% of GDP)',
'Government Effectiveness':'Government Effectiveness',
'Military Spending ':'Military Spending (% of GDP)',}
shortenFeatsReverse = {'Population':'Population',
'Area':'Area',
'GINI index':'GINI index',
'Happy Planet Index':'Happy Planet Index',
'Human Development Index':'Human Development Index',
'Sustainable Economic Ddevelopment Assessment (SEDA)':'SEDA',
'GDP ($ USD billions PPP)':'GDP ',
'GDP per capita in $ (PPP)':'GDP per capita ',
'Health Expenditure (% of GDP)':'Health Expenditure',
'Education Expenditure (% of GDP)':'Education Expenditure',
'Government Effectiveness':'Government Effectiveness',
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


d3.csv("data_processed_v1.csv").then(function(data) {
    
    dataset = data
    // console.log(dataset[155]);
    popScale= d3.scaleSqrt()
                .domain([d3.min(dataset.map((d)=>d.Population)),
                    d3.max(dataset.map((d)=>d.Population))])
                .range([8,12])
    countryColor = d3.scaleOrdinal(d3.schemeTableau10)
    .domain(dataset.map((d)=>d.Country).values());
    // drawInit()
    // drawInit();
    setTimeout(drawInit(), 100)
    
    // console.log("done")
    // draw1();
  });

function drawInit(){
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
    // .attr("background-color",'black')

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
            .attr('fill', (d)=>countryColor(d.Country))
            .attr('opacity', 1)
            .attr('cx', (d, i) => width/32)
            .attr('cy', (d, i) => height/2)
            // .attr('r', (d)=>0)
            // .transition().delay(100)
            .attr('r', (d)=>popScale(d.Population))
            .attr('id',(d)=>d['code'])
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
                .attr('x', (d, i) => width/32)
                .attr('y', (d, i) => height/2)   
    
    simulation  
            .force('charge', d3.forceManyBody().strength([5]))
            .force('forceX', d3.forceX(d => width/4).strength([0.6]))
            .force('forceY', d3.forceY(d => height/2).strength(0.04))
            .force('collide', d3.forceCollide(d => popScale(d.Population) + 2)).tick(100)
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

          // d3.select('#modalContent1').append('p').append('text').text(d.Country)
          let modalsvg = d3.select(".modalLeftContainer").append('svg')
                        .attr('width',600).attr('height',500)
          let leftScale = d3.scaleBand().rangeRound([0, 480])
                        leftScale.domain(Object.keys(shortenFeatsMapping))
          let bottomScale = d3.scaleBand().rangeRound([0, 600-50])
                        bottomScale.domain(['low','','high'])
          modalsvg.append('g').attr("class","leftAAxis")
                        .attr('font-size',8)
                          .attr('transform','translate(105,0)')
                          .call(d3.axisLeft(leftScale))
          modalsvg.append('g').attr("class","bottomAAxis")
                          .attr('font-size',8)
                            .attr('transform',`translate(105,${500 - 20})`)
                            .call(d3.axisBottom(bottomScale));


          let gdataset = dataset.filter((d_)=>d_.kgroup==d.kgroup)
          console.log(gdataset.length)


          allFeatures.map((feat,index)=>{
            // let feat = "sc_"+vfeat
          
            // .attr("width")
  
  // console.log(feat)
            let group1 = modalsvg.append('g')
              .attr('transform',"translate(110,20)")
              .attr('class',"circleGroup")
            let gx = d3.scaleLinear().domain([d3.min(gdataset.map((d_)=>parseFloat(d_[feat]))),
              d3.max(gdataset.map((d_)=>parseFloat(d_[feat])))]).range([20,460])
            // let gx = d3.scaleLinear().range([20,460])
            // gx.domain(d3.extent(gdataset.map((v)=>v[feat])))
            group1.selectAll('circle')
            .data(gdataset)
            .enter()
            .append('circle')
                .attr('fill', (d_)=>countryColor(d_.Country))
                .attr('opacity', 0.8)
                .attr('cx', (d_, i) =>0)
                .attr('cy', (d_, i) => leftScale(shortenFeatsReverse[feat]))
                .attr('r', (d_)=>0)
                .attr('stroke',(d_)=>{
                  if (d_.Country===d.Country){return '#EE6C12'}else{return 'none'}
                })
                .attr('stroke-width',1)
                .on('mouseover',(d_)=>{
                  d3.select("#tooltip")
                   .style('left', (d3.event.pageX + 10)+ 'px')
                    .style('top', (d3.event.pageY - 25) + 'px')
                    .style('display', 'inline-block')
                    .style("background", "#FFFCF6")
                    .html(`<strong>${d_.Country}</strong><br>
                          ${feat}: ${d_[feat]}`)

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
                .attr('r', (d_)=>5)
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
                            .data(["Property","Value","Cluster Mean"])
                            .enter().append("th")
                            .text(function(d_) { return d_; })
                            .style("border", "3px #F1D46D solid")
                            .style("padding", "5px")
                            .style("background-color", "#F1D46D")
                            .style("font-weight", "12")
                            .style("text-transform", "uppercase");

          
          
          rightBlock.append("tbody")
                            .selectAll("tr").data(Object.keys(shortenFeatsMapping))
                            .enter().append("tr")
                            .selectAll("td")
                            .data(function(ft){return [ft, 
                              d[shortenFeatsMapping[ft]],
                              parseInt(d3.mean(gdataset.map((d_)=>parseFloat(d_[shortenFeatsMapping[ft]]))))
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
                        
        }
        
        function mouseOverTex(d, i){

          // console.log('hi')
          d3.select(this)
              .transition('mouseover').duration(100)
              .attr('opacity', 1)
              // .attr('stroke-width', 5)
              // .attr('stroke', 'black')
         
             
            }
        function mouseOutTex(d, i){
        
              // d3.select('#tooltip').select("circle").remove()
      
              d3.select(this)
                  .transition('mouseout').duration(100)
                  .attr('opacity', 0.8)
                  .attr('stroke-width', 0)
          }

    function mouseOver(d, i){

        // console.log('hi')
        d3.select(this)
            .transition('mouseover').duration(100)
            .attr('opacity', 1)
            .attr('stroke-width', 5)
            .attr('stroke', 'black')
        
          d3.select("#tooltipMain")
          .style('left', (d3.event.pageX + 10)+ 'px')
          .style('top', (d3.event.pageY - 25) + 'px')
          .style('display', 'inline-block')
          .style("background", "#FFFCF6")
          .html(`<strong>${d.Country}</strong>`)



          }
    
    function mouseOut(d, i){
        
        // d3.select('#tooltip').select("circle").remove()

        d3.select(this)
            .transition('mouseout').duration(100)
            .attr('opacity', 0.8)
            .attr('stroke-width', 0)
            // d3.select("#tooltipMain").remove().exit();
            d3.select("#tooltipMain")
            .style('display', 'none');
            // d3.select("#tooltipMain").remove().exit();
    }

    d3.select("#myList1").selectAll("option")
      .data(dataset)
      .enter()
      .append('option')
      .text((d)=>d.Country)
      .attr('value',(d)=>(d.Country));
      d3.select("#myList1").on('change',()=>{
        // console.log(d3.select("#myList1").property('value'))
        let query = d3.select("#myList1").property('value')
        query = dataset.filter((d)=>d.Country==query)[0]['code']
        // console.log(query)
        d3.selectAll("#"+query).transition().duration(500)
        .attr('opacity', 1)
        .attr('stroke-width', 2)
        .attr('stroke', 'black')
        .transition().duration(500)
        .attr('opacity', 0.5)
        .attr('stroke-width', 15)
        .attr('stroke', 'red')
        .transition().duration(500)
        .attr('opacity', 1)
        .attr('stroke-width', 3)
        .attr('stroke', 'black')
        .transition().duration(500)
        .attr('opacity', 1)
        .attr('stroke-width', 15)
        .attr('stroke', 'red')
        .transition().duration(500)
        .attr('opacity', 1)
        .attr('stroke-width', 3)
        .attr('stroke', 'black')
   
      })
      
}
function getReportingMetric(){
  return clusterFeats[0]
}
function draw1Alt(){
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
    .force('forceX', d3.forceX(d => foci[d.kgroup].x).strength([0.005]))
    .force('forceY', d3.forceY(d =>  foci[d.kgroup].y ).strength([0.14]))
    .force('collide', d3.forceCollide(d => popScale(d.Population)+1 ))
    .alpha(1).alphaDecay([0.05])

//Reheat simulation and restart
simulation.restart()
let svg = d3.select("#vis")
    .select('svg');
    svg.selectAll(".annotations").remove().exit()
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
    // foci.map((d,i)=>{
    //   annotations.push()
    // })
    
  svg
  .append("g")
  .selectAll('.annotations')
  .data(foci)
  .enter()
  .append('text')
  .attr("class","annotations")
  .text((d,i)=>{
    var displayText = ''
    clusterFeats.map((ft)=>{
      displayText = displayText + `Mean ${shortenFeatsReverse[ft]}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft])))}`
    })
    var ft = clusterFeats[0]
    displayText = `Mean ${shortenFeatsReverse[ft]}: ${parseInt(d3.mean(dataset.filter((d_)=>d_.kgroup==i).map((d_)=>d_[ft])))}`
    return displayText})
  .attr("font-weight", 44)
    .attr("font-size", 0)
    .attr('fill','#352F44')
    .attr('font-family','Inter')
    .style("text-anchor", "middle")
    .attr('x', (d, i) => d.x+350)
    .attr('y', (d, i) => d.y) 
    .transition().delay((d,i)=>i*200)
    .attr("font-size", 12)
  
  
    // console.log(svg)
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
    .force('forceX', d3.forceX(d => width/4).strength([0.6]))
    .force('forceY', d3.forceY(d => height/2).strength(0.04))
    .force('collide', d3.forceCollide(d => popScale(d.Population) + 1)).tick(100)
    .alphaDecay([0.05])
    // let svg = d3.select("#vis")
   
    d3.selectAll(".annotations").remove().exit()
//Reheat simulation and restart
simulation.alpha(1).restart()
        
}
