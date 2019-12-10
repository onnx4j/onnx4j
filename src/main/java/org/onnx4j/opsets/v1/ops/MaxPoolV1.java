package org.onnx4j.opsets.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
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
public interface MaxPoolV1<T_TENSOR> extends OperatorV1 {

	public static final String OP_TYPE = "MaxPool";

	public static final String ATTR_AUTO_PAD = "auto_pad";

	public static final String ATTR_KERNEL_SHAPE = "kernel_shape";

	public static final String ATTR_PADS = "pads";

	public static final String ATTR_STRIDES = "strides";

	/**
	 * MaxPool consumes an input tensor X and applies max pooling across the
	 * tensor according to kernel sizes, stride sizes, and pad lengths. max
	 * pooling consisting of computing the max on all values of a subset of the
	 * input tensor according to the kernel size and downsampling the data into
	 * the output tensor Y for further processing.
	 * 
	 * @param data
	 *            Input data tensor from the previous operator; dimensions for
	 *            image case are (N x C x H x W), where N is the batch size, C
	 *            is the number of channels, and H and W are the height and the
	 *            width of the data. For non image case, the dimensions are in
	 *            the form of (N x C x D1 x D2 ... Dn), where N is the batch
	 *            size. Optionally, if dimension denotation is in effect, the
	 *            operation expects the input data tensor to arrive with the
	 *            dimension denotation of [DATA_BATCH, DATA_CHANNEL,
	 *            DATA_FEATURE, DATA_FEATURE ...].
	 * @param autoPad
	 *            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or
	 *            VALID. Where default value is NOTSET, which means explicit
	 *            padding is used. SAME_UPPER or SAME_LOWER mean pad the input
	 *            so that the output spatial size match the input.In case of odd
	 *            number add the extra padding at the end for SAME_UPPER and at
	 *            the beginning for SAME_LOWER. VALID mean no padding.
	 * @param kernelShape
	 *            The size of the kernel along each axis.
	 * @param pads
	 *            Padding for the beginning and ending along each spatial axis,
	 *            it can take any value greater than or equal to 0. The value
	 *            represent the number of pixels added to the beginning and end
	 *            part of the corresponding axis. `pads` format should be as
	 *            follow [x1_begin, x2_begin...x1_end, x2_end,...], where
	 *            xi_begin the number of pixels added at the beginning of axis
	 *            `i` and xi_end, the number of pixels added at the end of axis
	 *            `i`. This attribute cannot be used simultaneously with
	 *            auto_pad attribute. If not present, the padding defaults to 0
	 *            along start and end of each spatial axis.
	 * @param strides
	 *            Stride along each spatial axis.
	 * @return Output data tensor from average or max pooling across the input
	 *         tensor. Dimensions will vary based on various kernel, stride, and
	 *         pad sizes. Floor value of the dimension is used
	 */
	public abstract T_TENSOR maxpool(T_TENSOR data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides);

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
		return Outputs.wrap(node, this.maxpool(inputArray[0].getTensor(), autoPad, kernelShape, pads, strides));
	}

}
