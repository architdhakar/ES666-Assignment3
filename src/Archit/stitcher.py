import cv2
import numpy as np
import glob
import os
import random

class PanaromaStitcher:
    def __init__(self):
        self.base_width = 900  

    def resize_image(self, image):
    
        resized_image = cv2.resize(image, (450, 600))
        return resized_image
    
    def stitch(self, images, ratio=0.75, reprojThresh=3.0, showMatches=False):
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None

        (matches, H, status) = M
        self.last_H = H

        # Warp the second image (imageA) into the perspective of the first image (imageB)
        max_height = max(imageA.shape[0], imageB.shape[0])
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], max_height))

       
        overlap_width = imageB.shape[1] // 2  # Width for feathering
        for x in range(overlap_width):
            alpha = np.exp(-np.square(x / overlap_width))
            result[0:imageB.shape[0], x] = cv2.addWeighted(result[0:imageB.shape[0], x], alpha, imageB[:, x], 1 - alpha, 0)


        # Place non-overlapping part of imageB
        result[0:imageB.shape[0], 0:imageB.shape[1] - overlap_width] = imageB[:, :imageB.shape[1] - overlap_width]

        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        return result

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        (kps, features) = sift.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh, maxIters=8000):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
           
            (H, status) = self.computeHomographyRANSAC(ptsA, ptsB, reprojThresh, maxIters)
            return (matches, H, status)

        print("Insufficient matches found")
        return None

    def computeHomography(self, ptsA, ptsB):
        A = []
        for i in range(ptsA.shape[0]):
            x, y = ptsA[i][0], ptsA[i][1]
            u, v = ptsB[i][0], ptsB[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))
        H /= H[2, 2]
        return H

    def computeHomographyRANSAC(self, ptsA, ptsB, reprojThresh, maxIters):
        bestH = None
        bestInliers = 0
        bestStatus = None

        for i in range(maxIters):
            idx = np.random.choice(len(ptsA), 4, replace=False)
            ptsA_sample, ptsB_sample = ptsA[idx], ptsB[idx]

            H = self.computeHomography(ptsA_sample, ptsB_sample)
            ptsA_proj = cv2.perspectiveTransform(ptsA.reshape(-1, 1, 2), H).reshape(-1, 2)

            distances = np.linalg.norm(ptsB - ptsA_proj, axis=1)
            status = distances < reprojThresh
            inliers = np.sum(status)

            if inliers > bestInliers:
                bestInliers = inliers
                bestH = H
                bestStatus = status

        print(f"Best homography found with {bestInliers} inliers out of {len(ptsA)} points")
        return bestH, bestStatus

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

    def make_panaroma_for_images_in(self, path, showMatches=False):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')

        if len(all_images) < 2:
            print("Need at least two images to stitch.")
            return None, []

        images = [self.resize_image(cv2.imread(img)) for img in all_images if cv2.imread(img) is not None]
        stitched_images = images

        while len(stitched_images) > 1:
            next_level_images = []

            for i in range(0, len(stitched_images) - 1, 2):
                stitched_pair = self.stitch((stitched_images[i], stitched_images[i + 1]), showMatches=showMatches)
                if stitched_pair is not None:
                    next_level_images.append(stitched_pair)
                else:
                    print("Stitching failed for a pair, skipping.")

            if len(stitched_images) % 2 == 1:
                next_level_images.append(stitched_images[-1])

            stitched_images = next_level_images

        return stitched_images[0] if stitched_images else None, []
