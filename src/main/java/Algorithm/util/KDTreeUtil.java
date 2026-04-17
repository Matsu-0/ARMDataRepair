package Algorithm.util;

import java.util.*;

public class KDTreeUtil {

    private static class Node {
        private double[] point;
        private Node left;
        private Node right;

        private Node(double[] point) {
            this.point = point;
        }
    }

    private Node root;

    public KDTreeUtil(double[][] points) {
        root = buildTree(points, 0);
    }

    private Node buildTree(double[][] points, int depth) {
        if (points.length == 0) {
            return null;
        }
        int axis = depth % points[0].length;
        Arrays.sort(points, Comparator.comparingDouble(a -> a[axis]));
        int mid = points.length / 2;
        Node node = new Node(points[mid]);
        node.left = buildTree(Arrays.copyOfRange(points, 0, mid), depth + 1);
        node.right = buildTree(Arrays.copyOfRange(points, mid + 1, points.length), depth + 1);
        return node;
    }

    public double nearestNeighborDistance(double[] query) {
        return distance(query, nearestNeighbor(root, query, root.point));
    }

    public double[] nearestNeighbor(double[] query) {
        return nearestNeighbor(root, query, root.point);
    }

    private double[] nearestNeighbor(Node node, double[] query, double[] best) {
        if (node == null) {
            return best;
        }
        if (distance(node.point, query) < distance(best, query)) {
            best = node.point;
        }
        int axis = 0;
        if (node.left != null && node.right != null) {
            axis = query[axis] > node.point[axis] ? 1 : 0;
        }
        best = nearestNeighbor(axis == 0 ? node.left : node.right, query, best);
        if (axis == 0 || query[axis] - best[axis] < distance(node.point, query)) {
            best = nearestNeighbor(axis == 0 ? node.right : node.left, query, best);
        }
        return best;
    }

    public double[][] kNearestNeighbors(double[] query, int k) {
        PriorityQueue<double[]> pq = new PriorityQueue<>(k, (a, b) -> Double.compare(distance(b, query), distance(a, query)));
        kNearestNeighbors(root, query, pq, k);
        double[][] result = new double[k][];
        while (!pq.isEmpty()) {
            result[--k] = pq.poll();
        }
        return result;
    }

    private void kNearestNeighbors(Node node, double[] query, PriorityQueue<double[]> pq, int k) {
        if (node == null) {
            return;
        }
        double distance = distance(node.point, query);
        if (pq.size() < k || distance < distance(pq.peek(), query)) {
            pq.offer(node.point);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        int axis = 0;
        if (node.left != null && node.right != null) {
            axis = query[axis] > node.point[axis] ? 1 : 0;
        }
        kNearestNeighbors(axis == 0 ? node.left : node.right, query, pq, k);
        if (axis == 0 || query[axis] - pq.peek()[axis] < distance(node.point, query)) {
            kNearestNeighbors(axis == 0 ? node.right : node.left, query, pq, k);
        }
    }

    private double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

}
