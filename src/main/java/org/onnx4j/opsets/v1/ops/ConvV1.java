package org.onnx4j.opsets.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.v1.OperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * MaxPool-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#MaxPool-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface ConvV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "Conv";

	public static final String ATTR_AUTO_PAD = "auto_pad";

	public static final String ATTR_KERNEL_SHAPE = "kernel_shape";

	public static final String ATTR_PADS = "pads";

	public static final String ATTR_STRIDES = "strides";

	public static final String ATTR_GROUP = "group";

	public static final String ATTR_DILATIONS = "dilations";

	/**
	 * The convolution operator consumes an input tensor and a filter, and
	 * computes the output.
	 * 
	 * @param x
	 *            Input data tensor from previous layer; has size (N x C x H x
	 *            W), where N is the batch size, C is the number of channels,
	 *            and H and W are the height and width. Note that this is for
	 *            the 2D image. Otherwise the size is (N x C x D1 x D2 ... x
	 *            Dn). Optionally, if dimension denotation is in effect, the
	 *            operation expects input data tensor to arrive with the
	 *            dimension denotation of [DATA_BATCH, DATA_CHANNEL,
	 *            DATA_FEATURE, DATA_FEATURE ...].
	 * @param w
	 *            The weight tensor that will be used in the convolutions; has
	 *            size (M x C/group x kH x kW), where C is the number of
	 *            channels, and kH and kW are the height and width of the
	 *            kernel, and M is the number of feature maps. For more than 2
	 *            dimensions, the kernel shape will be (M x C/group x k1 x k2 x
	 *            ... x kn), where (k1 x k2 x ... kn) is the dimension of the
	 *            kernel. Optionally, if dimension denotation is in effect, the
	 *            operation expects the weight tensor to arrive with the
	 *            dimension denotation of [FILTER_OUT_CHANNEL,
	 *            FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].
	 *            X.shape[1] == (W.shape[1] * group) == C (assuming zero based
	 *            indices for the shape array). Or in other words
	 *            FILTER_IN_CHANNEL should be equal to DATA_CHANNEL.
	 * @param b
	 *            Optional 1D bias to be added to the convolution, has size of
	 *            M.
	 * @param autoPad
	 * @param dilations
	 * @param group
	 * @param kernelShape
	 * @param pads
	 * @param strides
	 * @return
	 */
	public abstract T_TENSOR conv(T_TENSOR x, T_TENSOR w, T_TENSOR b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// string (default is NOTSET)
		//
		String autoPad = attrs.getAttrValue(ATTR_AUTO_PAD, StringAttribute.class, "NOTSET");

		//
		// list of ints
		//
		List<Long> dilations = attrs.getAttrValue(ATTR_DILATIONS, IntsAttribute.class, null);

		//
		// int (default is 1)
		//
		Long group = attrs.getAttrValue(ATTR_GROUP, IntAttribute.class, 1L);

		//
		// list of ints (required)
		//
		List<Long> kernelShape = attrs.getAttrValue(ATTR_KERNEL_SHAPE, IntsAttribute.class, null);

		//
		// list of ints
		//
		List<Long> pads = attrs.getAttrValue(ATTR_PADS, IntsAttribute.class, null);

		//
		// list of ints
		//
		List<Long> strides = attrs.getAttrValue(ATTR_STRIDES, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.conv(inputArray[0].getTensor(), inputArray[1].getTensor(),
						((inputArray.length > 2) ? inputArray[2].getTensor() : null), autoPad, dilations, group,
						kernelShape, pads, strides));
	}

}
