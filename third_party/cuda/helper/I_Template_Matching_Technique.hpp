#ifndef WIZARD_I_TEMPLATE_MATCHING_TECHNIQUE_HPP
#define WIZARD_I_TEMPLATE_MATCHING_TECHNIQUE_HPP

//#include "opencv2/opencv.hpp"

//#include "wizard/Space.hpp"

//! Pure interface encapsulating an algorithm capable of matching a template against some kind of source.
/*!
 * Used primarily for strip-based tracking. Perhaps this could be moved out of Core::Strip if there is use elsewhere.
 *
 * This interface is written with the expectation that we change the template far more often than we change the source.
 */
class I_Template_Matching_Technique {
public:
    //! Deconstructor
    virtual ~I_Template_Matching_Technique() = default;

    //! Sets the source to use. Warm-up code ought to belong here. e.g. any pre-computation needed by CUDA.
    /*!
     * \param source            The image that we want to perform template matching against
     * \param templ_size        The size of the templates that will be used against this source. Typically this just adds padding around the source image.
     */
    virtual void set_source(cv::Mat source, Eigen::Vector2i templ_size) = 0;

    //! Match the given template against the source. Can throw an exception if the source has not been provided yet.
    /*!
     * \param templ             The template image that we want to match against the source.
     * \param location          The vector which will be populated with the peak location, assuming we got a successful match
     *
     * Returns true if the match was successful or false if
     * there is no match "good enough" (implementation-dependent, e.g. the cross-correlation peak was not high enough)
     */
    virtual bool match(cv::Mat templ, Eigen::Vector2i& location) = 0;

	virtual bool match_time_stat(cv::Mat templ, Eigen::Vector2i& location, double* datapreptime, double* calculationtime, double* maxfindingtime) = 0;
};

#endif // WIZARD_I_TEMPLATE_MATCHING_TECHNIQUE_HPP