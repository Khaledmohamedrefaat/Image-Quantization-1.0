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
    public struct edge
    {
        public int idx, u, v;
        public double cost;
        public edge(int idxx, int uu, int vv, double c)
        {
            idx = idxx;
            u = uu;
            v = vv;
            cost = c;
        }
    }


    /// <summary>
    /// Library of static functions that deal with images
    /// </summary>
    public class ImageOperations
    {
        // Built in Functions

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
        }
        public static void displayOutput(RGBPixel[,] ImageMatrix, PictureBox PicBox)
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
        // Team Made Variables

        public static List<RGBPixel> distinctColors = new List<RGBPixel>();
        public static int[,,] colVis = new int[260, 260, 260];
        public static long OOl = long.MaxValue;
        public static double OOd = double.MaxValue;

        // Team Made Functions

        public static List<RGBPixel> getDistinctColors(RGBPixel[,] ImageMatrix)
        {
            Console.WriteLine("Generating Distinct Colors ...");
            List<RGBPixel> distinct = new List<RGBPixel>();
            int height = GetHeight(ImageMatrix);
            int width = GetWidth(ImageMatrix);

            // Initialize colVis
            // Complexity --> O(1) Constant Time Complexity 
            for (int i = 0; i < 260; ++i)
                for (int j = 0; j < 260; ++j)
                    for (int k = 0; k < 260; ++k)
                        colVis[i, j, k] = -1;

            // Get Distinct And The Visited ID of each color (r, g, b)
            // Complexity --> O(height * width)
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    int r = ImageMatrix[i, j].red;
                    int g = ImageMatrix[i, j].green;
                    int b = ImageMatrix[i, j].blue;

                    if (colVis[r, g, b] == -1)
                    {
                        colVis[r, g, b] = distinct.Count;
                        distinct.Add(ImageMatrix[i, j]);
                    }
                }
            }
            Console.WriteLine("Distinct Colors Generated with size " + distinct.Count + " !");
            return distinct;
        }


        // Gets The Euclidian Distance Between Two Points
        // Complexity => O(1)
        public static double getDist(double r1, double g1, double b1, double r2, double g2, double b2)
        {
            double dr = r2 - r1;
            double dg = g2 - g1;
            double db = b2 - b1;
            double sqDist = dr * dr + dg * dg + db * db;
            return sqDist;
        }


        // EFFICIENT implementation of minimum spanning tree
        // Complexity => O(V^2), V: Numbers Of Vertices
        public static List<KeyValuePair<int, int>> prim(List<int> indices)
        {
            Console.WriteLine("Generating Minimum Spanning Tree ...");
            int sz = indices.Count;

            double[] dist = new double[distinctColors.Count + 9];         // Best Distance to reach the node
            int[] parent = new int[distinctColors.Count + 9];            // Parent Of Each Node
            int[] notVis = new int[sz + 9];            // Node not visited Till Now

            // Initializing arrays 
            for (int i = 0; i < distinctColors.Count + 9; ++i) dist[i] = OOd;
            for (int i = 0; i < sz; ++i) notVis[i] = indices[i];


            // Mark Source (0)
            dist[indices[0]] = 0;
            parent[indices[0]] = -1;
            int nxt = 0;
            int s = sz;
            double mn;

            // Prim's Algorithm For Computing The Minimum Spanning Tree 
            // Complexity --> O(V^2)

            while (nxt != -1)
            {                  // Complexity --> O(V)
                int u = notVis[nxt];
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                notVis[nxt] = notVis[--s];
                nxt = -1;
                mn = OOd;
                // Iterate Over each node trying to relax the current distance.
                for (int i = 0; i < s; ++i)
                {    // Complexity --> O(V)
                    int v = notVis[i];
                    int r2 = distinctColors[v].red;
                    int g2 = distinctColors[v].green;
                    int b2 = distinctColors[v].blue;
                    double d = getDist(r1, g1, b1, r2, g2, b2); // Implicitly knowing the cost of the edge ( Distance )

                    if (d < dist[v])
                    {           // Relaxation
                        dist[v] = d;
                        parent[v] = u;
                    }
                    if (dist[v] < mn)
                    {          // Choosing the best candidate for the next iteration
                        mn = dist[v];
                        nxt = i;
                    }
                }
            }

            // Return The Edges Used for building the minimum spanning tree
            List<KeyValuePair<int, int>> ret = new List<KeyValuePair<int, int>>();
            double mstSum = 0;

            // Complexity => O(sz)
            for (int i = 0; i < sz; ++i)
            {
                ret.Add(new KeyValuePair<int, int>(indices[i], parent[indices[i]]));
                int u = ret[i].Key;
                int v = ret[i].Value;
                if (v == -1) continue;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                double d = getDist(r1, g1, b1, r2, g2, b2);
                mstSum += d;
            }
            Console.WriteLine("Minimum Spanning Tree Generated with Cost : " + mstSum + " !");
            return ret;
        }

        // Graph Variables
        public static int[] head, nxt, to, from;
        public static double[] cost;
        public static int ne;

        // Dfs Variables
        public static bool[] vis, eVis, updVis;
        public static int[] compID, compSz, repComp, parent;
        public static double[] mnRepComp;
        public static RGBPixelD[] compCentroid;
        public static int numComps;

        // Initializing Graph/DFS Variables
        public static void init(int n)
        {
            Console.WriteLine("Initializing Graph ...");
            ne = 0;
            head = new int[n + 5];
            nxt = new int[n + 5];
            to = new int[n + 5];
            from = new int[n + 5];
            cost = new double[n + 5];
            repComp = new int[n + 5];
            mnRepComp = new double[n + 5];
            vis = new bool[n + 5];
            updVis = new bool[n + 5];
            compID = new int[n + 5];
            eVis = new bool[n + 5];
            compSz = new int[n + 5];
            compCentroid = new RGBPixelD[n + 5];
            // Complexity => O(n)
            for (int i = 0; i < n + 5; ++i) head[i] = -1;
            for (int i = 0; i < n + 5; ++i)
                compCentroid[i].red = compCentroid[i].blue = compCentroid[i].green = 0;
            numComps = 0;
            Console.WriteLine("Initializing Graph Done !");
        }

        // Graph Building Functions

        public static void addEdge(int f, int t, double c)
        {
            from[ne] = f;
            to[ne] = t;
            nxt[ne] = head[f];
            cost[ne] = c;
            head[f] = ne++;
        }
        public static void addBiEdge(int a, int b, double c)
        {
            addEdge(a, b, c);
            addEdge(b, a, c);
        }

        // Sorting Edges
        public static List<KeyValuePair<int, int>> sortEdges(List<KeyValuePair<int, int>> edgeList, int K)
        {
            Console.WriteLine("Sorting Edges ...");
            List<KeyValuePair<double, int>> sortedEdges = new List<KeyValuePair<double, int>>();
            for (int i = 0; i < edgeList.Count; ++i)
            {
                int u = edgeList[i].Key;
                int v = edgeList[i].Value;
                if (v == -1) continue;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                double dist = getDist(r1, g1, b1, r2, g2, b2);
                sortedEdges.Add(new KeyValuePair<double, int>(dist, i));
            }

            // Quick Sort => O(N log N)
            sortedEdges.Sort((x, y) => x.Key.CompareTo(y.Key));

            List<KeyValuePair<int, int>> newEdgeList = new List<KeyValuePair<int, int>>();
            for (int i = 0; i < sortedEdges.Count - K + 1; ++i)
                newEdgeList.Add(edgeList[sortedEdges[i].Value]);

            Console.WriteLine("Sorting Edges Done !");
            return newEdgeList;
        }

        // Graph construction
        public static void constructGraph(List<KeyValuePair<int, int>> edgeList)
        {
            Console.WriteLine("Constructing Graph...");
            int sz = edgeList.Count;
            // Complexity => O(sz)
            for (int i = 0; i < sz; ++i)
            {
                int u = edgeList[i].Key;
                int v = edgeList[i].Value;
                if (v == -1) continue;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                double dist = getDist(r1, g1, b1, r2, g2, b2);
                addBiEdge(u, v, dist);
            }
            Console.WriteLine("Constructing Graph Done !");
        }

        // Depth First Search
        // Complexity => O(distinct Colors)
        public static void dfs(int u)
        {
            vis[u] = true;
            compID[u] = numComps; // Mark u with current Component

            compSz[numComps]++;
            compCentroid[numComps].red += distinctColors[u].red;
            compCentroid[numComps].green += distinctColors[u].green;
            compCentroid[numComps].blue += distinctColors[u].blue;

            for (int k = head[u]; k != -1; k = nxt[k])
            {
                if (eVis[k] == true) continue;
                int v = to[k];
                if (vis[v] == true) continue;
                dfs(v);
            }
        }
        // Updates The Whole Component with the representitive
        // Complexity => O(distinct Colors)
        public static void dfsUpdate(int u)
        {
            updVis[u] = true;
            parent[u] = repComp[compID[u]];
            for (int k = head[u]; k != -1; k = nxt[k])
            {
                if (eVis[k] == true) continue;
                int v = to[k];
                if (updVis[v] == true) continue;
                dfsUpdate(v);
            }
        }

        // Choosing Best Representitive for each component
        // Palette generation by calculating the clusters centroids
        // Complexity => O(n), n : number of indices
        public static void computeRep(List<int> indices)
        {
            Console.WriteLine("Computing Representatives ...");
            for (int i = 1; i <= numComps; ++i)
            {
                repComp[i] = -1;
                mnRepComp[i] = OOd;
            }
            for (int i = 0; i < indices.Count; ++i)
            {
                int currID = compID[indices[i]];
                int r1 = distinctColors[indices[i]].red;
                int g1 = distinctColors[indices[i]].green;
                int b1 = distinctColors[indices[i]].blue;
                RGBPixelD centroid = compCentroid[currID];
                double r2 = centroid.red;
                double g2 = centroid.green;
                double b2 = centroid.blue;
                double dis = getDist(r1, g1, b1, r2, g2, b2);
                if (dis <= mnRepComp[currID])
                {
                    mnRepComp[currID] = dis;
                    repComp[currID] = indices[i];
                }
            }
            Console.WriteLine("Computing Representatives Done !");
        }

        public static void processComps(List<int> indices)
        {
            int n = indices.Count;
            Console.WriteLine("Starting DFS ...");
            // DFS From Each Component => O(distinct Colors)
            // Dfs is not nested, it is Amortized O(distinct Colors)
            for (int i = 0; i < n; ++i)
            {
                if (vis[indices[i]] == false)
                {
                    vis[indices[i]] = true;
                    numComps++;
                    dfs(indices[i]);
                    compCentroid[numComps].red /= (double)compSz[numComps];
                    compCentroid[numComps].green /= (double)compSz[numComps];
                    compCentroid[numComps].blue /= (double)compSz[numComps];
                }
            }
            Console.WriteLine("DFS Done !");
            computeRep(indices);
        }

        // Extracting the K clusters
        // Complexity => O(n)
        public static void extractKclusters(List<KeyValuePair<int, int>> edgeList, int K)
        {
            int n = distinctColors.Count;
            init(2 * n + 30); // O(n)
            edgeList = sortEdges(edgeList, K); // O(n)
            constructGraph(edgeList); // O(n)
            List<int> initialIndices = new List<int>();
            for (int i = 0; i < distinctColors.Count; ++i)
                initialIndices.Add(i);
            processComps(initialIndices); // O(n)
        }

        public static double getMean(List<double> data, int l, int r)
        {
            double ret = 0;
            double sum = 0;
            double sz = data.Count;
            for (int i = l; i <= r; ++i)
                sum += data[i];
            ret = sum / sz;
            return ret;
        }

        public static double getStdDev(List<double> data, int l, int r)
        {
            double ret = 0;
            double mean = getMean(data, l, r);
            double sum = 0;
            double sz = data.Count;
            for (int i = l; i <= r; ++i)
                sum += (data[i] - mean) * (data[i] - mean);
            ret = Math.Sqrt(sum / (sz - 1));
            return ret;
        }

        // Complexity => O(K * D)
        public static KeyValuePair<List<KeyValuePair<int, int>>, int> detectKClusters(List<KeyValuePair<int, int>> edgeList)
        {
            Console.WriteLine("Starting Automatic Detection Of Clusters ...");
            // Sort the edges O(n log n)
            edgeList = sortEdges(edgeList, 1);
            List<double> data = new List<double>();
            double sum = 0;
            for (int i = 0; i < edgeList.Count; ++i)
            {
                int u = edgeList[i].Key;
                int v = edgeList[i].Value;
                if (v == -1) continue;
                int r1 = distinctColors[u].red;
                int g1 = distinctColors[u].green;
                int b1 = distinctColors[u].blue;
                int r2 = distinctColors[v].red;
                int g2 = distinctColors[v].green;
                int b2 = distinctColors[v].blue;
                double dist = getDist(r1, g1, b1, r2, g2, b2);
                sum += dist;
                data.Add(dist);
            }
            double sz = data.Count;
            double mean = sum / sz;
            double limit = 0.0001;
            int l = 0, r = data.Count - 1;
            double currStdDev = getStdDev(data, l, r);
            double MSTStdDev = currStdDev;
            double nextStdDev, reduction = OOd;
            double lastReduction = OOd;
            //var s = string.Format("{0:0.##}", MSTStdDev);
            //Console.WriteLi+ne(s);
            int K = 1;
            // Complexity O(K * D)
            while (l < r)
            {
                double diffmn = Math.Abs(data[l] - mean);
                double diffmx = Math.Abs(data[r] - mean);
                if (diffmn > diffmx)
                    sum -= data[l++];
                else
                    sum -= data[r--];

                sz--;
                mean = getMean(data, l, r);
                nextStdDev = getStdDev(data, l, r);
                reduction = MSTStdDev - nextStdDev;

                if ((double)Math.Abs(reduction - lastReduction) <= limit)
                {
                    if (diffmn > diffmx) l--;
                    else r++;
                    break;
                }
                currStdDev = nextStdDev;
                lastReduction = reduction;
                ++K;
            }
            KeyValuePair<List<KeyValuePair<int, int>>, int> ret = new KeyValuePair<List<KeyValuePair<int, int>>, int>();
            List<KeyValuePair<int, int>> edges = new List<KeyValuePair<int, int>>();
            for (int i = l; i <= r; ++i)
                edges.Add(edgeList[i]);
            ret = new KeyValuePair<List<KeyValuePair<int, int>>, int>(edges, K);
            Console.WriteLine("The Best Cluster Number is : " + K + " !");
            return ret;
        }



        // Mapping the original colors to the palette colors
        // Complexity => O(h * w)
        public static RGBPixel[,] mapColors(RGBPixel[,] ImageMatrix)
        {
            int height = GetHeight(ImageMatrix);
            int width = GetWidth(ImageMatrix);
            RGBPixel[,] mappedMatrix = new RGBPixel[height, width];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    int r = ImageMatrix[i, j].red;
                    int g = ImageMatrix[i, j].green;
                    int b = ImageMatrix[i, j].blue;
                    int id = colVis[r, g, b];
                    int comp = compID[id];
                    int rep = repComp[comp];
                    mappedMatrix[i, j].red = distinctColors[rep].red;
                    mappedMatrix[i, j].green = distinctColors[rep].green;
                    mappedMatrix[i, j].blue = distinctColors[rep].blue;
                }
            }
            return mappedMatrix;
        }


        // Complexity => O(V^2 * Log K)
        public static void betterExtractKCluster(List<KeyValuePair<int, int>> MSTedgeList, int K)
        {
            Console.WriteLine("Better Extracting K Clusters ...");
            parent = new int[MSTedgeList.Count + 30];
            int Q = 0;
            if (K == -1) K = detectKClusters(MSTedgeList).Value;
            List<int> nodes = new List<int>();
            for (int i = 0; i < distinctColors.Count; ++i)
                nodes.Add(i);
            int iterations = 1;
            while (Q != K)
            {
                Console.WriteLine("Currently Working in the Iteration No. : " + iterations + " ...");
                List<KeyValuePair<int, int>> edgeList;
                if (iterations == 1)
                    edgeList = MSTedgeList;
                else
                    edgeList = prim(nodes);
                List<edge> edges = new List<edge>();
                init(2 * distinctColors.Count + 30);
                constructGraph(edgeList);
                List<double> data = new List<double>();
                for (int i = 0; i < ne; i += 2)
                {
                    edges.Add(new edge(i, from[i], to[i], cost[i]));
                    data.Add(cost[i]);
                }
                double mean = getMean(data, 0, data.Count - 1);
                double stdDev = getStdDev(data, 0, data.Count - 1);
                int delEdges = 0;
                for (int i = 0; i < ne; i += 2)
                {
                    //if (cost[i] > mean + stdDev || cost[i] < mean - stdDev)
                    if (cost[i] > mean + stdDev)
                    {
                        eVis[i] = true;
                        eVis[i + 1] = true;
                        delEdges++;
                    }
                }
                Q = delEdges + 1;
                Console.WriteLine("Found " + Q + " Components.");
                if (Q <= K)
                {
                    edges.Sort((x, y) => x.cost.CompareTo(y.cost));
                    for (int i = edges.Count - 1; i >= 0 && Q != K; --i)
                        if (eVis[edges[i].idx] == false)
                        {
                            eVis[edges[i].idx] = true;
                            eVis[edges[i].idx + 1] = true;
                            ++Q;
                        }
                    processComps(nodes);
                    int n = nodes.Count;
                    for (int i = 0; i < n; ++i)
                    {
                        if (updVis[nodes[i]] == false)
                        {
                            updVis[nodes[i]] = true;
                            dfsUpdate(nodes[i]);
                        }
                    }
                }
                else
                {
                    processComps(nodes);
                    int n = nodes.Count;
                    for (int i = 0; i < n; ++i)
                    {
                        if (updVis[nodes[i]] == false)
                        {
                            updVis[nodes[i]] = true;
                            dfsUpdate(nodes[i]);
                        }
                    }
                    nodes = new List<int>();
                    for (int i = 1; i <= numComps; ++i)
                        nodes.Add(repComp[i]);
                }

                Console.WriteLine("The Iteration No. " + iterations++ + " is Done !");
            }
            Console.WriteLine("Better Extracting K Clusters Done !");
        }

        // Complexity => O(log(K))
        public static int getParent(int u)
        {
            if (parent[u] == u) return u;
            return parent[u] = getParent(parent[u]);
        }

        // Complexity => O(h * w * log(K))
        public static RGBPixel[,] betterMapColors(RGBPixel[,] ImageMatrix)
        {
            int height = GetHeight(ImageMatrix);
            int width = GetWidth(ImageMatrix);
            RGBPixel[,] mappedMatrix = new RGBPixel[height, width];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    int r = ImageMatrix[i, j].red;
                    int g = ImageMatrix[i, j].green;
                    int b = ImageMatrix[i, j].blue;
                    int id = colVis[r, g, b];
                    id = getParent(id);
                    int comp = compID[id];
                    int rep = repComp[comp];
                    mappedMatrix[i, j].red = distinctColors[rep].red;
                    mappedMatrix[i, j].green = distinctColors[rep].green;
                    mappedMatrix[i, j].blue = distinctColors[rep].blue;
                }
            }
            return mappedMatrix;
        }

        public static RGBPixel[,] processImage(RGBPixel[,] ImageMatrix, int K)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            distinctColors = getDistinctColors(ImageMatrix);
            List<int> initialIndices = new List<int>();
            for (int i = 0; i < distinctColors.Count; ++i)
                initialIndices.Add(i);
            List<KeyValuePair<int, int>> edgeList = prim(initialIndices);
            if (K == -1)
                K = detectKClusters(edgeList).Value;
            //extractKclusters(edgeList, K);
            //ImageMatrix = mapColors(ImageMatrix);
            betterExtractKCluster(edgeList, K);
            ImageMatrix = betterMapColors(ImageMatrix);
            //dbgDistinct(ImageMatrix);
            //dbgComps();
            //dbgRep();
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            elapsedMs /= 1000;
            Console.WriteLine("The Code finished Successfully and ran in : " + elapsedMs + " seconds.");
            return ImageMatrix;
        }

        public static void dbgDistinct(RGBPixel[,] ImageMatrix)
        {
            Console.WriteLine("Debugging Matrix ...");
            int height = GetHeight(ImageMatrix);
            int width = GetWidth(ImageMatrix);

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    Console.Write(colVis[ImageMatrix[i, j].red, ImageMatrix[i, j].green, ImageMatrix[i, j].blue] + " ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine("Debugging Matrix Done !");
        }

        public static void dbgComps()
        {
            Console.WriteLine("Debugging Comps ... ");
            for (int i = 0; i < distinctColors.Count; ++i)
            {
                Console.WriteLine("Node " + i + " => Comp " + compID[i] + " => R " + distinctColors[i].red + " => G " + distinctColors[i].green + " => B " + distinctColors[i].blue);
            }
            Console.WriteLine("Debugging Comps Done !");
        }
        public static void dbgRep()
        {
            Console.WriteLine("Debugging Rep ... ");
            for (int i = 1; i <= numComps; ++i)
            {
                int rep = repComp[i];
                RGBPixelD center = compCentroid[i];
                Console.Write("Comp " + i + "=> Rep " + repComp[i] + " => R " + distinctColors[rep].red + " => G " + distinctColors[rep].green + " => B " + distinctColors[rep].blue);
                Console.WriteLine(" => R " + center.red + " => G " + center.green + " => B " + center.blue);
            }
            Console.WriteLine("Debugging Rep Done ! ");
        }
    }
}
