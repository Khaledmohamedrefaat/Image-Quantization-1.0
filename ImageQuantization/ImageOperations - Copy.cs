using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Windows.Forms;
using System.Drawing.Imaging;
///Algorithms Project
///Intelligent Scissors
///

namespace ImageQuantization
{

    /// <summary>
    /// Holds the pixel color in 3 byte values: red, green and blue
    /// </summary>
    public struct RGBPixel
    {
        public byte red, green, blue;
    }
    public struct RGBPixelD
    {
        public double red, green, blue;
    }
    
  
    /// <summary>
    /// Library of static functions that deal with images
    /// </summary>
    public class ImageOperations
    {
        /// <summary>
        /// Open an image and load it into 2D array of colors (size: Height x Width)
        /// </summary>
        /// <param name="ImagePath">Image file path</param>
        /// <returns>2D array of colors</returns>
        public static RGBPixel[,] OpenImage(string ImagePath)
        {
            Bitmap original_bm = new Bitmap(ImagePath);
            int Height = original_bm.Height;
            int Width = original_bm.Width;

            RGBPixel[,] Buffer = new RGBPixel[Height, Width];

            unsafe
            {
                BitmapData bmd = original_bm.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.ReadWrite, original_bm.PixelFormat);
                int x, y;
                int nWidth = 0;
                bool Format32 = false;
                bool Format24 = false;
                bool Format8 = false;

                if (original_bm.PixelFormat == PixelFormat.Format24bppRgb)
                {
                    Format24 = true;
                    nWidth = Width * 3;
                }
                else if (original_bm.PixelFormat == PixelFormat.Format32bppArgb || original_bm.PixelFormat == PixelFormat.Format32bppRgb || original_bm.PixelFormat == PixelFormat.Format32bppPArgb)
                {
                    Format32 = true;
                    nWidth = Width * 4;
                }
                else if (original_bm.PixelFormat == PixelFormat.Format8bppIndexed)
                {
                    Format8 = true;
                    nWidth = Width;
                }
                int nOffset = bmd.Stride - nWidth;
                byte* p = (byte*)bmd.Scan0;
                for (y = 0; y < Height; y++)
                {
                    for (x = 0; x < Width; x++)
                    {
                        if (Format8)
                        {
                            Buffer[y, x].red = Buffer[y, x].green = Buffer[y, x].blue = p[0];
                            p++;
                        }
                        else
                        {
                            Buffer[y, x].red = p[2];
                            Buffer[y, x].green = p[1];
                            Buffer[y, x].blue = p[0];
                            if (Format24) p += 3;
                            else if (Format32) p += 4;
                        }
                    }
                    p += nOffset;
                }
                original_bm.UnlockBits(bmd);
            }

            return Buffer;
        }

        public static List<RGBPixel> getDistinctColors(RGBPixel[,] ImageMatrix){
            List<RGBPixel> distinct = new List<RGBPixel>();
            int height = GetHeight(ImageMatrix);
            int width  = GetWidth(ImageMatrix);
            colVis = new int[260, 260, 260];

            for(int i = 0; i < height; ++i){
                for(int j = 0; j < width; ++j){
                    int r = ImageMatrix[i, j].red;
                    int g = ImageMatrix[i, j].green;
                    int b = ImageMatrix[i, j].blue;

                    if(colVis[r, g, b] == -1){
                        colVis[r, g, b] = distinct.Count;
                        distinct.Add(ImageMatrix[i, j]);
                    }

                }
            }

            return distinct;
        }

        public static long getDist(int r1, int g1, int b1, int r2, int g2, int b2){
            long dr = r2 - r1;
            long dg = g2 - g1;
            long db = b2 - b1;
            long sqDist = dr * dr + dg * dg + db * db;
            return sqDist;
        }

        public static List<KeyValuePair<int, int>> prim(){
            long OO = 10000000000;
            int sz = distinctColors.Count;
            long[] dist = new long[sz + 9];
            int[] parent = new int[sz + 9];
            int[] notVis = new int[sz + 9];

            for (int i = 0; i < sz + 9; ++i) dist[i] = OO;
            for (int i = 0; i < sz + 9; ++i) notVis[i] = i;

            dist[0] = 0;
            parent[0] = -1;
            int nxt = 0;
            int s = sz;
            long mn;
            while(nxt != -1){
                int u = notVis[nxt];
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                notVis[nxt] = notVis[--s];
                nxt = -1;
                mn = OO;
                for (int i = 0; i < s; ++i){
                    int v = notVis[i];
                    int r2 = distinctColors[v].red;
                    int g2 = distinctColors[v].green;
                    int b2 = distinctColors[v].blue;
                    long d = getDist(r1, g1, b1, r2, g2, b2);
                    if (d < dist[v]){
                        dist[v] = d;
                        parent[v] = u;
                    }
                    if(dist[v] < mn){
                        mn = dist[v];
                        nxt = i;
                    }
                }
            }

            List<KeyValuePair<int, int>> ret = new List<KeyValuePair<int, int>>();
            for (int i = 0; i < sz; ++i)
                ret.Add(new KeyValuePair<int, int>(i, parent[i]));
            return ret;
        }

        public static int[] head, nxt, to, compID, compSz, compSum, from, repComp;
        public static int[,,] colVis;
        public static double[] mnRepComp;
        public static double[] cost;
        public static int ne, numComps;
        public static bool[] vis, eVis;
        public static List<RGBPixel> distinctColors;

        public static void initialize(int n){
            for (int i = 0; i < n + 5; ++i)
                head[i] = -1;
            ne = 0;
        }
        public static void addEdge(int f, int t, double c){
            from[ne] = f;
            to[ne] = t;
            nxt[ne] = head[f];
            cost[ne] = c;
            head[f] = ne++;
        }
        public static void addBiEdge(int a, int b, double c){
            addEdge(a, b, c);
            addEdge(b, a, c);
        }
        public static double getMean(List<int> nodes){
            double ret = 0;
            double sz = nodes.Count;
            for(int i = 0; i < sz; ++i){
                int flat = distinctColors[nodes[i]].red * 256 * 256 + distinctColors[nodes[i]].green * 256 + distinctColors[nodes[i]].blue;
                ret += flat;
            }
            ret /= sz;
            return ret;
        }
        public static double getStdDev(List<int> nodes)
        {
            double ret = 0;
            double sz = nodes.Count;
            double mean = getMean(nodes);
            for (int i = 0; i < sz; ++i){
                int flat = distinctColors[nodes[i]].red * 256 * 256 + distinctColors[nodes[i]].green * 256 + distinctColors[nodes[i]].blue;
                double diff = flat - mean;
                ret += diff * diff;
            }
            ret /= sz;
            return Math.Sqrt(ret);
        }

        public static void constructGraph(List<KeyValuePair<int, int>> edgeList){
            int sz = edgeList.Count;
            for(int i = 0; i < sz; ++i){
                int u = edgeList[i].Key;
                int v = edgeList[i].Value;
                if (v == -1) continue;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                long sqDist = getDist(r1, g1, b1, r2, g2, b2);
                double dist = Math.Sqrt(sqDist);
                addBiEdge(u, v, dist);
            }
        }

        public static void dfs(int u){
            vis[u] = true;
            compID[u] = numComps;
            compSz[numComps]++;
            compSum[numComps] += distinctColors[u].red * 256 * 256 + distinctColors[u].green * 256 + distinctColors[u].blue;
            for(int k = head[u]; k != -1; k = nxt[k]){
                if (eVis[k] == true) continue;
                int v = to[k];
                if (vis[v] == true) continue;
                dfs(v);
            }
        }

        public static void extractKclustersBonus(List<KeyValuePair<int, int>> edgeList, int K){
            long OO = 10000000000;
            int n = distinctColors.Count;
            head = new int[n + 5];
            nxt = new int[n + 5];
            to = new int[n + 5];
            from = new int[n + 5];
            repComp = new int[n + 5];
            mnRepComp = new double[n + 5];
            initialize(n);
            int Q = 1;
            List<int> nodes = new List<int>();
            List<int> delEdges = new List<int>();
            for (int i = 0; i < n; ++i)
                nodes.Add(i);
            constructGraph(edgeList);
            double mean, stdDev;
            while (Q != K){
                Q = 1;
                mean = getMean(nodes);
                stdDev = getStdDev(nodes);
                int nSz = nodes.Count;
                eVis = new bool[nSz + 5];
                for (int e = 0; e < ne; ++e){
                    if (cost[e] > mean + stdDev){
                        eVis[e] = true;
                        delEdges.Add(e);
                    }
                }
                vis = new bool[nSz + 5];
                compID = new int[nSz + 5];
                numComps = 0;
                for(int i = 0; i < nSz; ++i){
                    if(vis[nodes[i]] == false){
                        vis[nodes[i]] = true;
                        numComps++;
                        dfs(nodes[i]);
                    }
                }
                Q = numComps;
                if(Q > K){
                    for(int i = 0; i < numComps; ++i){
                        repComp[i] = -1;
                        mnRepComp[i] = OO;
                    }
                    for(int i = 0; i < nSz; ++i){
                        int curr = nodes[i], currID = compID[curr];
                        double flatCurr = distinctColors[curr].red * 256 * 256 + distinctColors[curr].green * 256 + distinctColors[curr].blue;
                        double center = 1.0 * compSum[currID] / compSz[currID];
                        double dx = flatCurr - center;
                        double dis = dx * dx;
                        if(dis < mnRepComp[currID]){
                            mnRepComp[currID] = dis;
                            repComp[currID] = i;
                        }
                    }
                    initialize(numComps + 2);
                    for(int i = 0; i < delEdges.Count; ++i){
                        int e = delEdges[i];
                        int f = from[e];
                        int cmpU = compID[f];
                        int t = to[e];
                        int cmpV = compID[t];
                        int u = repComp[cmpU];
                        int v = repComp[cmpV];
                        int r1 = distinctColors[u].red;
                        int g1 = distinctColors[u].green;
                        int b1 = distinctColors[u].blue;
                        int r2 = distinctColors[v].red;
                        int g2 = distinctColors[v].green;
                        int b2 = distinctColors[v].blue;
                        long sqDist = getDist(r1, g1, b1, r2, g2, b2);
                        double dist = Math.Sqrt(sqDist);
                        addBiEdge(u, v, dist);
                    }
                }
                else{
                    List<KeyValuePair<int, double>> edges = new List<KeyValuePair<int, double>>();
                    for (int e = 0; e < ne; ++e){
                        if (eVis[e] == true) continue;
                        double d = cost[e];
                        edges.Add(new KeyValuePair<int, double>(e, d));
                    }
                    edges.Sort((x, y) => x.Value.CompareTo(y.Value));
                    for(int i = edges.Count - 1; i >= 0 && Q < K; --i, ++Q)
                        eVis[edges[i].Key] = true;
                    vis = new bool[nSz + 5];
                    compID = new int[nSz + 5];
                    numComps = 0;
                    for (int i = 0; i < nSz; ++i)
                    {
                        if (vis[nodes[i]] == false)
                        {
                            vis[nodes[i]] = true;
                            numComps++;
                            dfs(nodes[i]);
                        }
                    }
                    for (int i = 0; i < numComps; ++i)
                    {
                        repComp[i] = -1;
                        mnRepComp[i] = OO;
                    }
                    for (int i = 0; i < nSz; ++i)
                    {
                        int curr = nodes[i], currID = compID[curr];
                        double flatCurr = distinctColors[curr].red * 256 * 256 + distinctColors[curr].green * 256 + distinctColors[curr].blue;
                        double center = 1.0 * compSum[currID] / compSz[currID];
                        double dx = flatCurr - center;
                        double dis = dx * dx;
                        if (dis < mnRepComp[currID])
                        {
                            mnRepComp[currID] = dis;
                            repComp[currID] = i;
                        }
                    }
                }
            }

        }
        public static void extractKclusters(List<KeyValuePair<int, int>> edgeList, int K)
        {
            int n = distinctColors.Count;
            long OO = 10000000000;
            head = new int[n + 5];
            nxt = new int[n + 5];
            to = new int[n + 5];
            from = new int[n + 5];
            repComp = new int[n + 5];
            mnRepComp = new double[n + 5];
            initialize(n);
            List<KeyValuePair<double, int>> sortedEdges = new List<KeyValuePair<double, int>>();
            for (int i = 0; i < edgeList.Count; ++i)
            {
                int u = edgeList[i].Key;
                int v = edgeList[i].Value;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                long sqDist = getDist(r1, g1, b1, r2, g2, b2);
                double dist = Math.Sqrt(sqDist);
                sortedEdges.Add(new KeyValuePair<double, int>(dist, i));
            }
            sortedEdges.Sort();
            List<KeyValuePair<int, int>> newEdgeList = new List<KeyValuePair<int, int>>();
            int Q = 1;
            for (int i = 0; i < sortedEdges.Count - K + 1; ++i)
                newEdgeList.Add(edgeList[sortedEdges[i].Value]);
            constructGraph(newEdgeList);
            vis = new bool[n + 5];
            compID = new int[n + 5];
            numComps = 0;
            for (int i = 0; i < n; ++i)
            {
                if (vis[i] == false)
                {
                    vis[i] = true;
                    numComps++;
                    dfs(i);
                }
            }

            for (int i = 0; i < numComps; ++i)
            {
                repComp[i] = -1;
                mnRepComp[i] = OO;
            }
            for (int i = 0; i < n; ++i)
            {
                int curr = i, currID = compID[curr];
                double flatCurr = distinctColors[curr].red * 256 * 256 + distinctColors[curr].green * 256 + distinctColors[curr].blue;
                double center = 1.0 * compSum[currID] / compSz[currID];
                double dx = flatCurr - center;
                double dis = dx * dx;
                if (dis < mnRepComp[currID])
                {
                    mnRepComp[currID] = dis;
                    repComp[currID] = i;
                }

            }
        }
        public static RGBPixel[,] processImage(RGBPixel[,] ImageMatrix){
            distinctColors = getDistinctColors(ImageMatrix);
            List<KeyValuePair<int, int>> edgeList = prim();
            dbgMST(edgeList);
            dbgColors();
            extractKclusters(edgeList, 3);
            int height = GetHeight(ImageMatrix);
            int width = GetWidth(ImageMatrix);
            int[,] ImageDist = new int[height, width];
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j){
                    ImageDist[i, j] = colVis[ImageMatrix[i, j].red, ImageMatrix[i, j].green, ImageMatrix[i, j].blue];
                    int repColor = repComp[compID[ImageDist[i, j]]];
                    ImageMatrix[i, j].red = distinctColors[repColor].red;
                    ImageMatrix[i, j].blue = distinctColors[repColor].blue;
                    ImageMatrix[i, j].green = distinctColors[repColor].green;
                }
            return ImageMatrix;
        }
        public static void dbgColors(){
            Console.WriteLine("-----------------------");
            Console.WriteLine("The Size Of Distinct Colors is : " + distinctColors.Count);
            Console.WriteLine("-----------------------");
        }
        public static void dbgMST(List<KeyValuePair<int, int>> edgeList)
        {
            Console.WriteLine("-----------------------");
            double sum = 0;
            Console.WriteLine("The Mst Edge List : ");
            for (int i = 0; i < edgeList.Count; ++i)
            {
                Console.WriteLine(edgeList[i].Key + " " + edgeList[i].Value);
                if (edgeList[i].Value != -1)
                {
                    int u = edgeList[i].Key;
                    int v = edgeList[i].Value;
                    int r1 = distinctColors[u].red;
                    int g1 = distinctColors[u].green;
                    int b1 = distinctColors[u].blue;
                    int r2 = distinctColors[v].red;
                    int g2 = distinctColors[v].green;
                    int b2 = distinctColors[v].blue;
                    long sqDist = getDist(r1, g1, b1, r2, g2, b2);
                    double dist = Math.Sqrt(sqDist);
                    sum += dist;
                }
            }
            Console.WriteLine("The Mst Edge sum : " + sum);
            Console.WriteLine("-----------------------");
        }
        /// <summary>
        /// Get the height of the image 
        /// </summary>
        /// <param name="ImageMatrix">2D array that contains the image</param>
        /// <returns>Image Height</returns>
        public static int GetHeight(RGBPixel[,] ImageMatrix)
        {
            return ImageMatrix.GetLength(0);
        }

        /// <summary>
        /// Get the width of the image 
        /// </summary>
        /// <param name="ImageMatrix">2D array that contains the image</param>
        /// <returns>Image Width</returns>
        public static int GetWidth(RGBPixel[,] ImageMatrix)
        {
            return ImageMatrix.GetLength(1);
        }

        /// <summary>
        /// Display the given image on the given PictureBox object
        /// </summary>
        /// <param name="ImageMatrix">2D array that contains the image</param>
        /// <param name="PicBox">PictureBox object to display the image on it</param>
        public static void DisplayImage(RGBPixel[,] ImageMatrix, PictureBox PicBox)
        {
            // Create Image:
            //==============
            int Height = ImageMatrix.GetLength(0);
            int Width = ImageMatrix.GetLength(1);

            Bitmap ImageBMP = new Bitmap(Width, Height, PixelFormat.Format24bppRgb);

            unsafe
            {
                BitmapData bmd = ImageBMP.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.ReadWrite, ImageBMP.PixelFormat);
                int nWidth = 0;
                nWidth = Width * 3;
                int nOffset = bmd.Stride - nWidth;
                byte* p = (byte*)bmd.Scan0;
                for (int i = 0; i < Height; i++)
                {
                    for (int j = 0; j < Width; j++)
                    {
                        p[2] = ImageMatrix[i, j].red;
                        p[1] = ImageMatrix[i, j].green;
                        p[0] = ImageMatrix[i, j].blue;
                        p += 3;
                    }

                    p += nOffset;
                }
                ImageBMP.UnlockBits(bmd);
            }
            PicBox.Image = ImageBMP;
            processImage(ImageMatrix);
        }


       /// <summary>
       /// Apply Gaussian smoothing filter to enhance the edge detection 
       /// </summary>
       /// <param name="ImageMatrix">Colored image matrix</param>
       /// <param name="filterSize">Gaussian mask size</param>
       /// <param name="sigma">Gaussian sigma</param>
       /// <returns>smoothed color image</returns>
        public static RGBPixel[,] GaussianFilter1D(RGBPixel[,] ImageMatrix, int filterSize, double sigma)
        {
            int Height = GetHeight(ImageMatrix);
            int Width = GetWidth(ImageMatrix);

            RGBPixelD[,] VerFiltered = new RGBPixelD[Height, Width];
            RGBPixel[,] Filtered = new RGBPixel[Height, Width];

           
            // Create Filter in Spatial Domain:
            //=================================
            //make the filter ODD size
            if (filterSize % 2 == 0) filterSize++;

            double[] Filter = new double[filterSize];

            //Compute Filter in Spatial Domain :
            //==================================
            double Sum1 = 0;
            int HalfSize = filterSize / 2;
            for (int y = -HalfSize; y <= HalfSize; y++)
            {
                //Filter[y+HalfSize] = (1.0 / (Math.Sqrt(2 * 22.0/7.0) * Segma)) * Math.Exp(-(double)(y*y) / (double)(2 * Segma * Segma)) ;
                Filter[y + HalfSize] = Math.Exp(-(double)(y * y) / (double)(2 * sigma * sigma));
                Sum1 += Filter[y + HalfSize];
            }
            for (int y = -HalfSize; y <= HalfSize; y++)
            {
                Filter[y + HalfSize] /= Sum1;
            }

            //Filter Original Image Vertically:
            //=================================
            int ii, jj;
            RGBPixelD Sum;
            RGBPixel Item1;
            RGBPixelD Item2;

            for (int j = 0; j < Width; j++)
                for (int i = 0; i < Height; i++)
                {
                    Sum.red = 0;
                    Sum.green = 0;
                    Sum.blue = 0;
                    for (int y = -HalfSize; y <= HalfSize; y++)
                    {
                        ii = i + y;
                        if (ii >= 0 && ii < Height)
                        {
                            Item1 = ImageMatrix[ii, j];
                            Sum.red += Filter[y + HalfSize] * Item1.red;
                            Sum.green += Filter[y + HalfSize] * Item1.green;
                            Sum.blue += Filter[y + HalfSize] * Item1.blue;
                        }
                    }
                    VerFiltered[i, j] = Sum;
                }

            //Filter Resulting Image Horizontally:
            //===================================
            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                {
                    Sum.red = 0;
                    Sum.green = 0;
                    Sum.blue = 0;
                    for (int x = -HalfSize; x <= HalfSize; x++)
                    {
                        jj = j + x;
                        if (jj >= 0 && jj < Width)
                        {
                            Item2 = VerFiltered[i, jj];
                            Sum.red += Filter[x + HalfSize] * Item2.red;
                            Sum.green += Filter[x + HalfSize] * Item2.green;
                            Sum.blue += Filter[x + HalfSize] * Item2.blue;
                        }
                    }
                    Filtered[i, j].red = (byte)Sum.red;
                    Filtered[i, j].green = (byte)Sum.green;
                    Filtered[i, j].blue = (byte)Sum.blue;
                }

            return Filtered;
        }


    }
}
