#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"


struct Patch {
    cv::Mat1d texture;
    int row;
    int col;
};


cv::Mat read_image(const std::string lr,
                   const std::string ref_id) {
    cv::Mat image;

    std::string image_name = "../../stereo_imgs/rectified_"+lr+"_"+ref_id+".png";
    image = cv::imread( image_name, 1 );
    if ( !image.data )
        std::cout << "No image data" << std::endl;

    return image;
}

void write_image(const cv::Mat & left,const std::string name) {

    std::string image_name = "../../output_imgs/"+name+".png";
    bool error = cv::imwrite(
                     image_name,
                     left);
    if (!error)
        std::cout << "not capable to write image" << std::endl;
}

cv::Mat apply_sobel_filter(const cv::Mat &left ) {

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    cv::Sobel( left, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( left, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    cv::Mat left_grad;

    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, left_grad );

    return left_grad;
}

std::vector<cv::Point2i> mask(const cv::Mat & input, cv::Mat out) {

    std::vector<cv::Point2i> points;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int val = (int)input.at<cv::Vec3b>(i,j)[0];
            if (val>8) {
                cv::Point2i p(j,i);
                points.push_back(p);
                out.at<uchar>(i,j) = 255;
            }
        }
    }
    return points;
}

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

cv::Mat1d convet_to_double(const cv::Mat &input) {
    cv::Mat1d out = cv::Mat1d::zeros(input.size());

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            out.at<double>(i,j) = (int)input.at<cv::Vec3b>(i,j)[0];
        }
    }

    return out;
}

std::vector<Patch> get_points_to_be_matched(
    const cv::Mat &left,
    const std::vector<cv::Point2i> &left_edge_points,
    const int semi_stencil) {

    std::vector<Patch> to_be_matched;

    for (const auto &p : left_edge_points) {

        const int i = p.y;
        const int j = p.x;

        int local_size = 2*semi_stencil+1;

        cv::Rect local_roi(j-semi_stencil,i-semi_stencil,
                           local_size,local_size);

        //std::cout << "i = " << i << ", j = " << j << std::endl;
        // std::cout << "local_roi = " << local_roi << std::endl;
        // std::cout << "left.size() = " << left.size() << std::endl;

        cv::Mat tmp0 = left(local_roi);
        cv::Mat1d tmp1 = convet_to_double(tmp0);

        cv::Scalar tempVal = cv::mean( tmp1 );
        double mean = tempVal.val[0];

        tmp1 -= mean;

        double min, max;
        cv::minMaxLoc(tmp1, &min, &max);

        tmp1/=max;

        Patch ptc;
        ptc.row = i;
        ptc.col = j;
        if (mean > 1e-10) {

            tmp1.copyTo(ptc.texture);

            //ptc.texture.clone(tmp1);//to_be_matched.push_back(tmp1);
        } else {
            cv::Mat1d z = cv::Mat1d::zeros(
                                        local_size,local_size);
            z.copyTo(ptc.texture);
            //to_be_matched.push_back();
        }
        to_be_matched.push_back(ptc);
    }

    return to_be_matched;
}

std::vector<std::vector<Patch>> get_matching_candidates(
    const cv::Mat &right,
    const cv::Rect &left_roi,
    const int semi_stencil){

    std::vector<std::vector<Patch>> matching_candidates;

    for (int i = 0; i < left_roi.height ; i++) {
        std::vector<Patch> stride;
        for (int j = 0; j < left_roi.width ; j++) {

            int row = left_roi.y + i;
            int col = left_roi.x + j;

            // std::cout << "row = " << row
            //            << ", col = " << col << std::endl;

            int local_size = 2*semi_stencil+1;

            cv::Rect local_roi(col-semi_stencil,row-semi_stencil,
                               local_size,local_size);
            //std::cout << local_roi << std::endl;

            cv::Mat tmp0 = right(local_roi);

            cv::Mat1d tmp1 = convet_to_double(tmp0);

            cv::Scalar tempVal = cv::mean( tmp1 );
            double mean = tempVal.val[0];

            tmp1 -= mean;

            double min, max;
            cv::minMaxLoc(tmp1, &min, &max);
            tmp1/=max;

            Patch ptc;
            ptc.row = row;
            ptc.col = col;

            if (mean > 1e-10) {
                tmp1.copyTo(ptc.texture);

                //stride.push_back(tmp1);
            } else {
                cv::Mat1d z = cv::Mat1d::ones(
                                            local_size,local_size);
                z.copyTo(ptc.texture);
                //stride.push_back(cv::Mat1d::ones(
                //                            local_size,local_size));
            }
            stride.push_back(ptc);

        }
        matching_candidates.push_back(stride);
    }

    return matching_candidates;
}

int main(int argc, char** argv )
{

    cv::Mat left = read_image("left","03");
    cv::Mat right = read_image("right","03");

    cv::Mat left_grad = apply_sobel_filter(left);
    cv::Mat right_grad = apply_sobel_filter(right);

    cv::Mat mask_left = cv::Mat::zeros(left_grad.size(), CV_8UC1);
    std::vector<cv::Point2i> left_edge_points =
        mask(left_grad,mask_left);

    cv::Mat mask_right = cv::Mat::zeros(right_grad.size(), CV_8UC1);
    std::vector<cv::Point2i> right_edge_points =
        mask(right_grad,mask_right);
    //cv::Mat mask_right = mask(right_grad);
    cv::Rect left_roi =  boundingRect(mask_left);

    cv::Mat tmp = cv::Mat::zeros(left_grad.size(), CV_8UC1);

    int semi_stencil = 3;

    std::vector<Patch> to_be_matched =
        get_points_to_be_matched(left,left_edge_points,semi_stencil);

    std::vector<std::vector<Patch>> matching_candidates =
         get_matching_candidates(right,left_roi,semi_stencil);

    cv::Mat1b matching_points = cv::Mat1b::zeros(left.size());

    for (const auto & left_patch : to_be_matched){
        // std::cout << "left_patch.row = " << left_patch.row
        //           << ", left_patch.col = " << left_patch.col << std::endl;

        int i = left_patch.row - left_roi.y;

        // std::cout << "left_patch.row = " << left_patch.row
        //           << ", right_patch.id = " << i << ", of: "<< matching_candidates.size() << std::endl;

        std::vector<double> cost;

        for (int j = 0; j < left_roi.width ; j++) {
            cost.push_back(cv::norm(left_patch.texture - matching_candidates[i][j].texture));
        }

        auto min_it = std::min_element(cost.begin(), cost.end());
        int min_id  = (int)std::distance(cost.begin(), min_it);

        matching_points.at<uchar>(matching_candidates[i][min_id].row,
            matching_candidates[i][min_id].col) = 255;


        std::cout << "right_patch.id = " << i << ", of: "<< matching_candidates.size() << std::endl;
    }
    //
    // int kont = 0;
    // for (const auto &p : left_edge_points){
    //     //to_be_matched[kont];
    //     const int i = p.y;
    //     const int j = p.x;
    //
    //     std::cout << "p = " << p << std::endl;
    //     for (int k = 0; k < left_roi.width ; k++) {
    //
    //         //int row = left_roi.y + i;
    //         int col = left_roi.x + k;
    //         int roi_id = i - left_roi.y;
    //
    //         std::cout << "i = " << i << ", j = " << j << std::endl;
    //
    //         // std::cout << cv::norm(to_be_matched[kont] -
    //         //                       matching_candidates[roi_id][k]) << std::endl;
    //     }

        // for (int k = 0; k < left_roi.width ; k++) {
        //
        //     //int row = left_roi.y + i;
        //     int col = left_roi.x + k;
        //     int roi_id = i - left_roi.y;
        //
        //     std::cout << "i = " << i << ", j = " << j << std::endl;
        //
        //     // std::cout << cv::norm(to_be_matched[kont] -
        //     //                       matching_candidates[roi_id][k]) << std::endl;
        // }



    //     kont++;
    // }






    /*std::vector< std::vector<cv::Mat1d> > right_patches(
        right.rows,std::vector<cv::Mat1d>(right.cols));*/

    //i = 63, j = 202



    // for (int i = 0; i < left_roi.height ; i++){
    //     for (int j = 0; j < left_roi.width ; j++){
    /*for (int i = 63; i < 64 ; i++){
        for (int j = 202; j < 203 ; j++){


            int row = left_roi.y + i;
            int col = left_roi.x + i;

            int local_size = 2*semi_stencil+1;

            cv::Rect local_roi(row-semi_stencil,col-semi_stencil,
                local_size,local_size);

            std::cout << local_roi << std::endl;

            cv::Mat tmp0 = right(local_roi);

            cv::Mat1d tmp1 = convet_to_double(tmp0);

            cv::Scalar tempVal = cv::mean( tmp1 );
            double mean = tempVal.val[0];

            tmp1 -= mean;

            double min, max;
            cv::minMaxLoc(tmp1, &min, &max);
            tmp1/=max;



            if (mean>1e-10) {
                cv::Mat_<double> m;
                tmp1.copyTo(m);
                right_patches[i][j].clone(m);
            } else {}
            //     //std::cout << "tmp0.size() = "<< tmp0.size() << std::endl;
            //     //cv::Mat1d uno = cv::Mat1d::zeros(11,11);
            //     // cv::Mat1d uno = cv::Mat::ones(
            //     //     tmp1.size(),CV_64F);
            //     //right_patches[i,j] = uno;
            // }


            // cv::cvtColor(tmp0,tmp1,CV_GRAY2RGB);
            // tmp1.convertTo(tmp1, CV_64F);
            // //B.copyTo(tmp);

            //std::cout << "i = " << row << ", j = " << col << std::endl;
            //std::cout << local_roi << std::endl;
            //std::cout << "tmp1" << tmp1 << std::endl;

        }

    }*/

    //std::cout << "left_roi = " << left_roi << std::endl;

    //std::string left_type =  type2str( left_grad.type() );
    //std::cout << "left_type = "<< left_type << std::endl;
    write_image(mask_left,"mask_left");
    write_image(matching_points,"matching_points");

    //std::cout << image << std::endl;

    return 0;
}
