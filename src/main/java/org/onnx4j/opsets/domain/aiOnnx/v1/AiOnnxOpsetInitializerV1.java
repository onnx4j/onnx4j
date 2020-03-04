/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.domain.aiOnnx.v1;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.domain.aiOnnx.AiOnnxOpsetInitializer;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AbsV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AddV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConcatV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConstantV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.DivV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.DropoutV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.GatherV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.IdentityV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.LeakyReluV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MatMulV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReduceMaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReluV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ShapeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SigmoidV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SoftmaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SqueezeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SubV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SumV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.TransposeV1;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.UnsqueezeV1;

/**
 * Default ONNX Operator Set in version 1
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOpsetInitializerV1 extends AiOnnxOpsetInitializer {

	public abstract AbsV1 getAbsV1();

	// public abstract PadV1 getPadV1();

	public abstract MatMulV1 getMatMulV1();

	public abstract IdentityV1 getIdentityV1();

	public abstract ArgMaxV1 getArgMaxV1();

	public abstract DivV1 getDivV1();

	public abstract ReshapeV1 getReshapeV1();

	public abstract AddV1 getAddV1();

	public abstract MaxPoolV1 getMaxPoolV1();

	public abstract ReluV1 getReluV1();

	public abstract ConvV1 getConvV1();

	public abstract ConstantV1 getConstantV1();

	public abstract ImageScalerV1 getImageScalerV1();

	public abstract BatchNormalizationV1 getBatchNormalizationV1();

	public abstract LeakyReluV1 getLeakyReluV1();

	public abstract MulV1 getMulV1();

	public abstract ConcatV1 getConcatV1();

	public abstract DropoutV1 getDropoutV1();

	public abstract AveragePoolV1 getAveragePoolV1();

	public abstract CastV1 getCastV1();

	public abstract GatherV1 getGatherV1();

	public abstract SubV1 getSubV1();

	public abstract SumV1 getSumV1();

	public abstract SigmoidV1 getSigmoidV1();

	public abstract SoftmaxV1 getSoftmaxV1();

	public abstract SqueezeV1 getSqueezeV1();

	public abstract UnsqueezeV1 getUnsqueezeV1();

	public abstract ReduceMaxV1 getReduceMaxV1();

	public abstract TransposeV1 getTransposeV1();

	public abstract ShapeV1 getShapeV1();

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOpsetInitializer.super.initializeOperators();
		operators.put(AbsV1.OP_TYPE, this.getAbsV1());
		// operators.put(PadV1.OP_TYPE, this.getPadV1());
		operators.put(MatMulV1.OP_TYPE, this.getMatMulV1());
		operators.put(IdentityV1.OP_TYPE, this.getIdentityV1());
		operators.put(ArgMaxV1.OP_TYPE, this.getArgMaxV1());
		// 20191024
		operators.put(DivV1.OP_TYPE, this.getDivV1());
		// 20191025
		operators.put(ReshapeV1.OP_TYPE, this.getReshapeV1());
		operators.put(AddV1.OP_TYPE, this.getAddV1());
		operators.put(MaxPoolV1.OP_TYPE, this.getMaxPoolV1());
		// 20191026
		operators.put(ReluV1.OP_TYPE, this.getReluV1());
		operators.put(ConvV1.OP_TYPE, this.getConvV1());
		operators.put(ConstantV1.OP_TYPE, this.getConstantV1());
		// 20191108
		operators.put(LeakyReluV1.OP_TYPE, this.getLeakyReluV1());
		// 20191111
		operators.put(BatchNormalizationV1.OP_TYPE, this.getBatchNormalizationV1());
		operators.put(ImageScalerV1.OP_TYPE, this.getImageScalerV1());
		// 20191115
		operators.put(MulV1.OP_TYPE, this.getMulV1());
		// 20191119
		operators.put(ConcatV1.OP_TYPE, this.getConcatV1());
		// 20191120
		operators.put(DropoutV1.OP_TYPE, this.getDropoutV1());
		operators.put(AveragePoolV1.OP_TYPE, this.getAveragePoolV1());
		// 20191216
		operators.put(CastV1.OP_TYPE, this.getCastV1());
		// 20191225
		operators.put(GatherV1.OP_TYPE, this.getGatherV1());
		// 20191227
		operators.put(SubV1.OP_TYPE, this.getSubV1());
		// 20191230
		operators.put(SumV1.OP_TYPE, this.getSumV1());
		// 20191231
		operators.put(SigmoidV1.OP_TYPE, this.getSigmoidV1());
		operators.put(SoftmaxV1.OP_TYPE, this.getSoftmaxV1());
		// 20200108
		operators.put(SqueezeV1.OP_TYPE, this.getSqueezeV1());
		// 20200110
		operators.put(UnsqueezeV1.OP_TYPE, this.getUnsqueezeV1());
		// 20200113
		operators.put(ReduceMaxV1.OP_TYPE, this.getReduceMaxV1());
		// 20200122
		operators.put(TransposeV1.OP_TYPE, this.getTransposeV1());
		operators.put(ShapeV1.OP_TYPE, this.getShapeV1());
		return operators;
	}

}